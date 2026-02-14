import torch

from ModelUniPhy import UniPhyModel


def _as_dt_seq(dt_context, bsz, t_in, device, dtype):
    if isinstance(dt_context, (float, int)):
        return torch.full((bsz, t_in), float(dt_context), device=device, dtype=dtype)
    if isinstance(dt_context, torch.Tensor):
        if dt_context.ndim == 0:
            return dt_context.to(device=device, dtype=dtype).expand(bsz, t_in)
        if dt_context.ndim == 1:
            return dt_context.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, t_in)
        if dt_context.ndim == 2:
            return dt_context.to(device=device, dtype=dtype)
    return torch.full((bsz, t_in), float(dt_context), device=device, dtype=dtype)


def _decode_step(model, z_step):
    y = model.decoder(z_step.unsqueeze(1))
    return y[:, 0]


def full_serial_inference(model, x_context, dt_context, dt_list):
    bsz, t_in = x_context.shape[0], x_context.shape[1]
    device = x_context.device
    dtype = torch.float64

    z_all = model.encoder(x_context)
    states = model._init_states()
    dt_ctx = _as_dt_seq(dt_context, bsz, t_in, device, dtype)

    for t in range(t_in):
        z_in = z_all[:, t]
        dt_t = dt_ctx[:, t]
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_in, h_next, flux_next = block.forward_step(z_in, h_prev, dt_t, flux_prev)
            states[i] = (h_next, flux_next)

    z_curr = model.encoder(x_context[:, -1].unsqueeze(1))[:, 0]

    preds = []
    for dt_k in dt_list:
        dt_k = dt_k.to(device=device, dtype=dtype)
        new_states = []
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_curr, h_next, flux_next = block.forward_step(z_curr, h_prev, dt_k, flux_prev)
            new_states.append((h_next, flux_next))
        states = new_states
        pred = _decode_step(model, z_curr)
        z_curr = model.encoder(pred.unsqueeze(1))[:, 0]
        preds.append(pred)

    return torch.stack(preds, dim=1)


def check_precision_robustness():
    print("=" * 60)
    print("Double Precision Verification (float64)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depth=2,
        patch_size=(4, 2),
        img_height=32,
        img_width=32,
        sde_mode="det",
    ).to(device=device, dtype=torch.float64)

    model.eval()

    bsz, t_len, channels, height, width = 2, 8, 4, 32, 32
    x = torch.randn(bsz, t_len, channels, height, width, device=device, dtype=torch.float64)
    dt = torch.ones(bsz, t_len, device=device, dtype=torch.float64) * 0.1

    print("Running Test 1: Parallel vs Serial (Basic Forward)...")
    with torch.no_grad():
        out_parallel = model(x, dt)

    out_serial = []
    z_all = model.encoder(x)
    states = model._init_states()

    for t in range(t_len):
        z_t = z_all[:, t]
        dt_t = dt[:, t]
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt_t, flux_prev)
            states[i] = (h_next, flux_next)
        out_serial.append(_decode_step(model, z_t))

    out_serial = torch.stack(out_serial, dim=1)
    diff_1 = (out_parallel - out_serial).abs().max().item()
    print(f"Test 1 Max Diff: {diff_1:.2e}")

    print("-" * 60)
    print("Running Test 2: Optimized Rollout vs Full Serial...")

    t_in, t_pred = 8, 4
    x_context = torch.randn(bsz, t_in, channels, height, width, device=device, dtype=torch.float64)
    dt_context = 1.0
    dt_list = [torch.tensor(0.5, device=device, dtype=torch.float64) for _ in range(t_pred)]

    with torch.no_grad():
        out_optimized = model.forward_rollout(x_context, dt_context, dt_list)
        out_full_serial = full_serial_inference(model, x_context, dt_context, dt_list)

    diff_2 = (out_optimized - out_full_serial).abs().max().item()
    print(f"Test 2 Max Diff: {diff_2:.2e}")

    print("=" * 60)
    passed = diff_1 < 1e-10 and diff_2 < 1e-10
    print("RESULT: PASSED" if passed else "RESULT: FAILED")
    return passed


if __name__ == "__main__":
    ok = check_precision_robustness()
    raise SystemExit(0 if ok else 1)
