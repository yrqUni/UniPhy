import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

LOG_DIR = Path("/nfs/Agent_dt_check/logs")
RESULTS = {}


def _imports():
    from Model.UniPhy.ModelUniPhy import UniPhyBlock, UniPhyModel
    from Model.UniPhy.UniPhyOps import ComplexSVDTransform
    from Model.UniPhy.UniPhyOps import _compute_sde_scale, _safe_forcing
    from Model.UniPhy.UniPhyOps import complex_dtype_for

    return (
        UniPhyBlock,
        UniPhyModel,
        ComplexSVDTransform,
        _compute_sde_scale,
        _safe_forcing,
        complex_dtype_for,
    )


def record(name, passed, detail=""):
    RESULTS[name] = (passed, detail)
    status = "PASS" if passed else "FAIL"
    print(f"{status:4}  {name}  {detail}")


def make_tiny(device):
    _, UniPhyModel, _, _, _, _ = _imports()
    torch.manual_seed(42)
    return UniPhyModel(
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


def check_no_linalg(device):
    del device
    _, _, ComplexSVDTransform, _, _, _ = _imports()
    src = inspect.getsource(ComplexSVDTransform)
    has_solve = "linalg.solve" in src
    has_inv = "linalg.inv" in src
    record("no_linalg_solve", not has_solve, f"linalg.solve found={has_solve}")
    record("no_linalg_inv", not has_inv, f"linalg.inv found={has_inv}")


def check_no_dead_methods(device):
    del device
    _, _, ComplexSVDTransform, _, _, _ = _imports()
    has_biortho = hasattr(ComplexSVDTransform, "get_biorthogonal_pair")
    has_encode = hasattr(ComplexSVDTransform, "encode")
    has_decode = hasattr(ComplexSVDTransform, "decode")
    record("no_get_biorthogonal_pair", not has_biortho)
    record("no_encode_method", not has_encode)
    record("no_decode_method", not has_decode)


def check_no_combine_output(device):
    del device
    UniPhyBlock, _, _, _, _, _ = _imports()
    src = inspect.getsource(UniPhyBlock)
    has_combine = "_combine_output" in src
    record("no_combine_output", not has_combine)


def check_no_dead_init_params(device):
    del device
    UniPhyBlock, _, _, _, _, _ = _imports()
    sig = str(inspect.signature(UniPhyBlock.__init__))
    has_img_h = "img_height" in sig
    has_img_w = "img_width" in sig
    has_kernel = "kernel_size" in sig
    record("no_img_height_in_block_init", not has_img_h)
    record("no_img_width_in_block_init", not has_img_w)
    record("no_kernel_size_in_block_init", not has_kernel)


def check_get_matrix_formula(device):
    _, _, ComplexSVDTransform, _, _, _ = _imports()
    basis = ComplexSVDTransform(dim=32).to(device)
    optimizer = torch.optim.SGD(
        [basis.w_re, basis.w_im, basis.w_inv_re, basis.w_inv_im, basis.alpha_logit],
        lr=1e-3,
    )
    identity = torch.eye(32, dtype=torch.complex128, device=device)
    with torch.no_grad():
        w0, w_inv0 = basis.get_matrix(torch.complex128)
        residual_0 = float((w0 @ w_inv0 - identity).abs().max().item())
    for _ in range(100):
        x = torch.randn(8, 32, device=device)
        x = torch.complex(x, torch.randn_like(x))
        w, w_inv = basis.get_matrix(torch.complex64)
        h = basis.encode_with(x, w)
        x_rec = basis.decode_with(h, w_inv)
        loss = (x_rec - x).abs().pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        w1, w_inv1 = basis.get_matrix(torch.complex128)
        residual_100 = float((w1 @ w_inv1 - identity).abs().max().item())
        cond = float(torch.linalg.cond(w1).item())
    detail = (
        f"residual_0={residual_0:.2e} residual_100={residual_100:.2e} "
        f"cond={cond:.1f}"
    )
    record("get_matrix_returns_pair", True, detail)


def check_w_re_gradient(device):
    torch.manual_seed(0)
    _, _, ComplexSVDTransform, _, _, _ = _imports()
    basis = ComplexSVDTransform(dim=16).to(device)
    x_real = torch.randn(4, 16, device=device)
    x = torch.complex(x_real, torch.randn_like(x_real))
    w, w_inv = basis.get_matrix(torch.complex64)
    h = basis.encode_with(x, w)
    x_rec = basis.decode_with(h, w_inv)
    x_rec.real.sum().backward()
    for name in ["w_re", "w_im", "w_inv_re", "w_inv_im"]:
        p = getattr(basis, name)
        has_grad = p.grad is not None and p.grad.abs().max().item() > 0
        val = p.grad.abs().max().item() if has_grad else 0.0
        record(f"basis_grad_{name}", has_grad, f"{val:.3e}")


def check_model_train_gradient(device):
    model = make_tiny(device)
    model.train()
    x = torch.randn(1, 2, 4, 721, 1440, device=device)
    dt = torch.full((1, 2), 6.0, device=device)
    out = model.forward(x, dt)
    out.real.sum().backward()
    p = model.blocks[0].prop.basis.w_re
    has_grad = p.grad is not None and p.grad.abs().max().item() > 0
    val = p.grad.abs().max().item() if has_grad else 0.0
    record("model_train_w_re_gradient", has_grad, f"w_re.grad.max={val:.3e}")


def check_numerical_constants(device):
    del device
    _imports()
    src_ops = inspect.getsource(sys.modules["Model.UniPhy.UniPhyOps"])
    record(
        "eps_1e7_present",
        "eps=1e-7" in src_ops,
        "eps=1e-7 in _safe_forcing/_compute_sde_scale",
    )
    record(
        "alpha_logit_init_2",
        "tensor(2.0)" in src_ops,
        "alpha_logit initial value",
    )
    record("gate_min_001", "0.01" in src_ops, "gate_min in GlobalFluxTracker")
    record("gate_max_099", "0.99" in src_ops, "gate_max in GlobalFluxTracker")


def check_phi1_stability(device):
    _, _, _, _, _safe_forcing, _ = _imports()
    z_vals = torch.logspace(-10, -4, 50, dtype=torch.float64, device=device)
    max_err = 0.0
    for z in z_vals:
        z_f32 = z.float().to(torch.complex64)
        ref = (
            torch.expm1(z.to(torch.complex128)) / z.to(torch.complex128)
        ).abs().item()
        got = _safe_forcing(z_f32, torch.tensor(1.0, device=device)).abs().item()
        rel_err = abs(got - ref) / (ref + 1e-30)
        max_err = max(max_err, rel_err)
    record("phi1_stability", max_err < 1e-5, f"max_rel_err={max_err:.2e}")


def check_sde_scale_dt_zero(device):
    _, _, _, _compute_sde_scale, _, _ = _imports()
    base_noise = torch.ones(8, device=device)
    dt_small = [1e-6, 1e-5, 1e-4, 1e-3]
    scales_a = [
        float(
            _compute_sde_scale(
                torch.full((8,), -1.0, device=device),
                torch.tensor(dt, device=device),
                base_noise,
            ).max().item()
        )
        for dt in dt_small
    ]
    check_a = all(scales_a[i] < scales_a[i + 1] for i in range(3))
    check_a = check_a and scales_a[0] < 1e-2
    lam_vals = [-0.1, -1.0, -10.0, -100.0]
    scales_b = [
        float(
            _compute_sde_scale(
                torch.full((8,), lam, device=device),
                torch.tensor(1.0, device=device),
                base_noise,
            ).max().item()
        )
        for lam in lam_vals
    ]
    check_b = all(scales_b[i] > scales_b[i + 1] for i in range(3))
    scale_c = _compute_sde_scale(
        torch.zeros(8, device=device),
        torch.tensor(1.0, device=device),
        base_noise,
    )
    analytic = base_noise * torch.sqrt(torch.tensor(1.0, device=device))
    err_c = float((scale_c - analytic).abs().max().item())
    lam_re = torch.full((8,), -0.5, device=device)
    dt_vals = [0.5, 1.0, 2.0, 4.0]
    scales_d = [
        float(
            _compute_sde_scale(
                lam_re,
                torch.tensor(dt, device=device),
                base_noise,
            ).max().item()
        )
        for dt in dt_vals
    ]
    check_d = all(scales_d[i] < scales_d[i + 1] for i in range(3))
    passed = check_a and check_b and err_c < 1e-5 and check_d
    record("sde_scale_physics", passed, f"err_c={err_c:.2e}")


def check_forward_vs_step(device):
    torch.manual_seed(7)
    model = make_tiny(device)
    model.eval()
    x = torch.randn(1, 1, 4, 721, 1440, device=device)
    dt = torch.full((1, 1), 6.0, device=device)
    with torch.no_grad():
        _, _, _, _, _, complex_dtype_for = _imports()
        dtype = complex_dtype_for(torch.float32)
        z = model.encoder(x)[:, 0]
        states = model._init_states(1, device, dtype)
        h0, f0 = states[0]
        z_seq, h_seq, _ = model.blocks[0].forward(z.unsqueeze(1), h0, dt, f0)
        z_step, h_step, _ = model.blocks[0].forward_step(z, h0, dt[:, 0], f0)
    err = (z_seq[:, 0] - z_step).abs().max().item()
    err_h = (h_seq - h_step).abs().max().item()
    record("forward_vs_step_latent", err < 1e-4, f"max_err={err:.2e}")
    record("forward_vs_step_h_state", err_h < 1e-4, f"max_err={err_h:.2e}")


def check_dt_zero_mask(device):
    model = make_tiny(device)
    model.eval()
    x = torch.randn(1, 1, 4, 721, 1440, device=device)
    dt_zero = torch.zeros(1, 1, device=device)
    with torch.no_grad():
        out = model.forward(x, dt_zero)
        out_r = out.real if out.is_complex() else out
    err = (out_r - x).abs().max().item()
    record("dt_zero_identity", err < 1e-6, f"max_err={err:.2e}")


def check_negative_dt_raises(device):
    model = make_tiny(device)
    x = torch.randn(1, 1, 4, 721, 1440, device=device)
    try:
        model.forward(x, torch.tensor([[-1.0]], device=device))
        record("negative_dt_raises", False, "no exception raised")
    except ValueError:
        record("negative_dt_raises", True)


def check_encode_decode_identity(device):
    _, _, ComplexSVDTransform, _, _, _ = _imports()

    def run_case(basis, x, dtype):
        w, w_inv = basis.get_matrix(dtype)
        h = basis.encode_with(x, w)
        x_rec = basis.decode_with(h, w_inv)
        return float((x - x_rec).abs().max().item())

    torch.manual_seed(42)
    basis = ComplexSVDTransform(dim=32).to(device)
    x2 = torch.randn(4, 32, device=device)
    x2 = torch.complex(x2, torch.randn_like(x2))
    x4 = torch.randn(2, 3, 5, 32, device=device)
    x4 = torch.complex(x4, torch.randn_like(x4))
    x5 = torch.randn(2, 4, 3, 5, 32, device=device)
    x5 = torch.complex(x5, torch.randn_like(x5))
    with torch.no_grad():
        err_a = max(
            run_case(basis, x2, torch.complex64),
            run_case(basis, x4, torch.complex64),
            run_case(basis, x5, torch.complex64),
        )
        basis.w_re.copy_(torch.randn(32, 32, device=device) * 0.01)
        basis.w_im.copy_(torch.randn(32, 32, device=device) * 0.01)
        err_b = run_case(basis, x2, torch.complex64)
        basis.alpha_logit.copy_(torch.tensor(10.0, device=device))
        w, w_inv = basis.get_matrix(torch.complex128)
        identity = torch.eye(32, dtype=torch.complex128, device=device)
        err_c = float((w @ w_inv - identity).abs().max().item())
    passed = err_a < 1e-5 and err_b < 1e-4 and err_c < 1e-5
    detail = f"err_a={err_a:.2e} err_b={err_b:.2e} err_c={err_c:.2e}"
    record("encode_decode_identity", passed, detail)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")

    check_no_linalg(device)
    check_no_dead_methods(device)
    check_no_combine_output(device)
    check_no_dead_init_params(device)
    check_get_matrix_formula(device)
    check_w_re_gradient(device)
    check_model_train_gradient(device)
    check_numerical_constants(device)
    check_phi1_stability(device)
    check_sde_scale_dt_zero(device)
    check_forward_vs_step(device)
    check_dt_zero_mask(device)
    check_negative_dt_raises(device)
    check_encode_decode_identity(device)

    total = len(RESULTS)
    passed = sum(1 for v, _ in RESULTS.values() if v)
    failed = total - passed

    print(f"\nRESULTS: {passed}/{total} PASS  {failed}/{total} FAIL")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f'{"PASS" if v else "FAIL"}  {k}  {d}' for k, (v, d) in RESULTS.items()]
    lines.append(f"SUMMARY: {passed}/{total} PASS  {failed}/{total} FAIL")
    (LOG_DIR / "check_result.txt").write_text("\n".join(lines) + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
