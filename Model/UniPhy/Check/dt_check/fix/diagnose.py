import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))


def _imports():
    from Model.UniPhy.ModelUniPhy import (
        UniPhyBlock,
        UniPhyModel,
        _dt_is_zero,
        _expand_batch_mask,
    )
    from Model.UniPhy.UniPhyOps import ComplexSVDTransform, complex_dtype_for

    return (
        UniPhyBlock,
        UniPhyModel,
        _dt_is_zero,
        _expand_batch_mask,
        ComplexSVDTransform,
        complex_dtype_for,
    )


(
    UniPhyBlock,
    UniPhyModel,
    _dt_is_zero,
    _expand_batch_mask,
    ComplexSVDTransform,
    complex_dtype_for,
) = _imports()


LOG_DIR = "/nfs/Agent_dt_check/logs"
ORIGINAL_FORWARD = UniPhyBlock.forward


def _forward_no_checkpoint(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
    batch_size, steps, dim, height, width = x.shape
    x_flat = x.reshape(batch_size * steps, dim, height, width)
    x_flat = self._apply_spatial(x_flat)
    x = x_flat.reshape(batch_size, steps, dim, height, width)
    basis_dtype = complex_dtype_for(x.dtype)
    basis_w, basis_w_inv = self.prop.get_basis_matrices(basis_dtype)
    x_perm = x.permute(0, 1, 3, 4, 2)
    x_eigen = self.prop.basis.encode_with(x_perm, basis_w)
    x_mean = self.flux_pool(x_eigen)
    a_flux, x_flux = self.prop.flux_tracker.get_scan_operators(x_mean, dt_seq)
    from Model.UniPhy.PScan import pscan

    flux_seq = pscan(a_flux, x_flux).squeeze(-1)
    decay_seq = a_flux.squeeze(-1)
    decay_cum = torch.cumprod(decay_seq, dim=1)
    flux_seq = flux_seq + flux_prev.unsqueeze(1) * decay_cum
    source_seq, gate_seq = self.prop.flux_tracker.compute_output_seq(flux_seq)
    flux_out = flux_seq[:, -1]
    forcing = self.spatial_gate(gate_seq, source_seq, x_eigen)
    op_decay, op_forcing = self.prop.get_transition_operators_seq(dt_seq)
    op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(
        batch_size, steps, height, width, dim
    )
    op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(
        batch_size, steps, height, width, dim
    )
    u_t = forcing * op_forcing
    noise_seq = self._normalize_block_noise_seq(
        noise_seq, batch_size, steps, dim, height, width
    )
    u_t = u_t + self.prop.generate_stochastic_term_seq(
        u_t.shape,
        dt_seq,
        u_t.dtype,
        x_eigen,
        noise_seq=noise_seq,
    )
    h_prev_hw = h_prev.reshape(batch_size, height, width, dim)
    h_contrib_t0 = h_prev_hw * op_decay[:, 0]
    u_t0 = u_t[:, 0] + h_contrib_t0
    u_t = torch.cat([u_t0.unsqueeze(1), u_t[:, 1:]], dim=1)
    a_scan = op_decay.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    x_scan = u_t.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    u_out = pscan(a_scan, x_scan).reshape(
        batch_size, height, width, steps, dim
    ).permute(0, 3, 1, 2, 4)
    h_out = u_out[:, -1].reshape(batch_size * height * width, 1, dim)
    decoded = self._decode_sequence(u_out, basis_w_inv)
    decoded_flat = decoded.reshape(batch_size * steps, dim, height, width)
    decoded_with_residual_flat = self._apply_temporal_decode(decoded_flat)
    decoded_with_residual = decoded_with_residual_flat.reshape(
        batch_size, steps, dim, height, width
    )
    combined = decoded_with_residual
    zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
    combined = torch.where(zero_mask, x, combined)
    return combined, h_out, flux_out


def patched_forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
    if getattr(self, "_force_no_checkpoint", False):
        return _forward_no_checkpoint(self, x, h_prev, dt_seq, flux_prev, noise_seq)
    return ORIGINAL_FORWARD(self, x, h_prev, dt_seq, flux_prev, noise_seq)


UniPhyBlock.forward = patched_forward


def grad_value(param):
    if param.grad is None:
        return 0.0
    return float(param.grad.abs().max().item())


def has_grad(param):
    return grad_value(param) > 0.0


def check_basis_isolated(device):
    torch.manual_seed(0)
    basis = ComplexSVDTransform(dim=32).to(device)
    x_real = torch.randn(4, 32, device=device)
    x = torch.complex(x_real, torch.randn_like(x_real))
    w, w_inv = basis.get_matrix(torch.complex64)
    h = basis.encode_with(x, w)
    x_rec = basis.decode_with(h, w_inv)
    x_rec.real.sum().backward()
    return has_grad(basis.w_re), grad_value(basis.w_re)


def make_tiny(device):
    torch.manual_seed(42)
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=8,
        expand=2,
        depth=1,
        patch_size=[7, 15],
        img_height=721,
        img_width=1440,
        dt_ref=6.0,
        init_noise_scale=1e-4,
    ).to(device)
    return model


def check_model_eval_mode(device):
    model = make_tiny(device)
    model.eval()
    torch.set_grad_enabled(True)
    x = torch.randn(1, 2, 4, 721, 1440, device=device)
    dt = torch.full((1, 2), 6.0, device=device)
    out = model.forward(x, dt)
    out.real.sum().backward()
    param = model.blocks[0].prop.basis.w_re
    return has_grad(param), grad_value(param)


def check_model_train_mode(device):
    model = make_tiny(device)
    model.train()
    x = torch.randn(1, 2, 4, 721, 1440, device=device)
    dt = torch.full((1, 2), 6.0, device=device)
    out = model.forward(x, dt)
    out.real.sum().backward()
    param = model.blocks[0].prop.basis.w_re
    return has_grad(param), grad_value(param)


def check_model_train_no_checkpoint(device):
    model = make_tiny(device)
    model.train()
    for block in model.blocks:
        block._force_no_checkpoint = True
    x = torch.randn(1, 2, 4, 721, 1440, device=device)
    dt = torch.full((1, 2), 6.0, device=device)
    out = model.forward(x, dt)
    out.real.sum().backward()
    param = model.blocks[0].prop.basis.w_re
    return has_grad(param), grad_value(param)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r1_pass, r1_val = check_basis_isolated(device)
    r2_pass, r2_val = check_model_eval_mode(device)
    r3_pass, r3_val = check_model_train_mode(device)
    r4_pass, r4_val = check_model_train_no_checkpoint(device)
    print(
        f"D1 basis isolated (no model, no checkpoint): "
        f"{'GRAD' if r1_pass else 'ZERO'} ({r1_val:.2e})"
    )
    print(
        f"D2 model eval mode (no checkpoint):          "
        f"{'GRAD' if r2_pass else 'ZERO'} ({r2_val:.2e})"
    )
    print(
        f"D3 model train mode (with checkpoint):       "
        f"{'GRAD' if r3_pass else 'ZERO'} ({r3_val:.2e})"
    )
    print(
        f"D4 model train, no checkpoint forced:        "
        f"{'GRAD' if r4_pass else 'ZERO'} ({r4_val:.2e})"
    )
    if r1_pass and r2_pass and not r3_pass and r4_pass:
        verdict = "CAUSE_A"
        reason = "gradient present without checkpoint, zero with checkpoint"
    elif not r1_pass:
        verdict = "CAUSE_B"
        reason = "gradient zero even in isolated basis test, no checkpoint involved"
    elif r1_pass and r2_pass and not r3_pass and not r4_pass:
        verdict = "CAUSE_B"
        reason = "gradient zero even without checkpoint in full model"
    else:
        verdict = "INCONCLUSIVE"
        reason = (
            f"unexpected pattern: D1={r1_pass} D2={r2_pass} "
            f"D3={r3_pass} D4={r4_pass}"
        )
    print(f"VERDICT: {verdict}")
    print(f"REASON:  {reason}")
    result_path = Path(LOG_DIR) / "diagnose_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        f"VERDICT: {verdict}\n"
        f"REASON: {reason}\n"
        f"D1: {r1_pass} {r1_val:.2e}\n"
        f"D2: {r2_pass} {r2_val:.2e}\n"
        f"D3: {r3_pass} {r3_val:.2e}\n"
        f"D4: {r4_pass} {r4_val:.2e}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
