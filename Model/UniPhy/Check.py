import torch
from ModelUniPhy import UniPhyModel


def full_serial_inference(model, x_context, dt_context, dt_list):
    B, T_in = x_context.shape[0], x_context.shape[1]
    device = x_context.device

    z_all = model.encoder(x_context)
    dtype = z_all.dtype
    states = model._init_states(B, device, dtype)

    dt_ctx_val = float(dt_context) if isinstance(dt_context, (float, int)) else dt_context.item()

    for t in range(T_in):
        z_in = z_all[:, t]
        dt_t = torch.full((B,), dt_ctx_val, device=device, dtype=torch.float64)
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_in, h_next, flux_next = block.forward_step(
                z_in, h_prev, dt_t, flux_prev,
            )
            states[i] = (h_next, flux_next)

    x_last = x_context[:, -1]
    z_curr = model.encoder(x_last)

    preds = []
    for dt_k in dt_list:
        dt_step = torch.full(
            (B,), dt_k.item(), device=device, dtype=torch.float64,
        )
        new_states = []
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_curr, h_next, flux_next = block.forward_step(
                z_curr, h_prev, dt_step, flux_prev,
            )
            new_states.append((h_next, flux_next))
        states = new_states
        pred = model.decoder(z_curr)
        z_curr = model.encoder(pred)
        preds.append(pred)

    return torch.stack(preds, dim=1)


def check_precision_robustness():
    print("=" * 60)
    print("Double Precision Verification (float64)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=4, out_channels=4, embed_dim=64, depth=2,
        patch_size=(4, 2), img_height=32, img_width=32, sde_mode="det",
    ).to(device).double()

    model.eval()

    B, T, C, H, W = 2, 8, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device).double()
    dt = torch.ones(B, T, device=device).double() * 0.1

    print("Running Test 1: Parallel vs Serial (Basic Forward)...")
    with torch.no_grad():
        out_parallel = model(x, dt)

        z_all = model.encoder(x)
        dtype = z_all.dtype
        states = model._init_states(B, device, dtype)

        out_serial = []
        for t in range(T):
            z_t = z_all[:, t]
            dt_t = dt[:, t]
            for i, block in enumerate(model.blocks):
                h, f = states[i]
                z_t, h_n, f_n = block.forward_step(z_t, h, dt_t, f)
                states[i] = (h_n, f_n)
            out = model.decoder(z_t)
            out_serial.append(out)

        out_serial = torch.stack(out_serial, dim=1)

    diff_1 = (out_parallel - out_serial).abs().max().item()
    print(f"Test 1 Max Diff: {diff_1:.2e}")

    print("-" * 60)
    print("Running Test 2: Optimized Rollout vs Full Serial...")

    B, T_in, T_pred = 2, 8, 4
    x_context = torch.randn(B, T_in, C, H, W, device=device).double()
    dt_context = 1.0
    dt_list = [
        torch.tensor(0.5, device=device, dtype=torch.float64)
        for _ in range(T_pred)
    ]

    with torch.no_grad():
        out_optimized = model.forward_rollout(x_context, dt_context, dt_list)
        out_full_serial = full_serial_inference(
            model, x_context, dt_context, dt_list,
        )

    diff_2 = (out_optimized - out_full_serial).abs().max().item()
    print(f"Test 2 Max Diff: {diff_2:.2e}")

    print("=" * 60)
    passed = diff_1 < 1e-10 and diff_2 < 1e-10

    if passed:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")

    return passed


if __name__ == "__main__":
    success = check_precision_robustness()
    exit(0 if success else 1)
    