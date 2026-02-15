import torch

from ModelUniPhy import UniPhyModel


def _decode_step(model, z_step):
    y = model.decoder(z_step.unsqueeze(1))
    return y[:, 0]


def full_serial_inference(model, x_context, dt_context_step, dt_future_step):
    bsz, t_ctx = x_context.shape[0], x_context.shape[1]
    states = model._init_states()

    if t_ctx >= 2:
        for t in range(t_ctx - 1):
            x_t = x_context[:, t]
            dt_t = dt_context_step[:, t]
            z_t = model.encoder(x_t.unsqueeze(1))[:, 0]
            new_states = []
            for i, block in enumerate(model.blocks):
                h_prev, flux_prev = states[i]
                z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt_t, flux_prev)
                new_states.append((h_next, flux_next))
            states = new_states

    z_curr = model.encoder(x_context[:, -1].unsqueeze(1))[:, 0]
    preds = []
    steps = int(dt_future_step.shape[1])
    for k in range(steps):
        dt_k = dt_future_step[:, k]
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

    bsz, k_len, channels, height, width = 2, 9, 4, 32, 32
    data = torch.randn(bsz, k_len, channels, height, width, device=device, dtype=torch.float64)
    x_input = data[:, :-1]
    dt_step = torch.ones(bsz, x_input.shape[1], device=device, dtype=torch.float64) * 0.1

    with torch.no_grad():
        out_parallel = model(x_input, dt_step)

    out_serial = []
    states = model._init_states()
    for t in range(x_input.shape[1]):
        x_t = x_input[:, t]
        dt_t = dt_step[:, t]
        z_t = model.encoder(x_t.unsqueeze(1))[:, 0]
        new_states = []
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt_t, flux_prev)
            new_states.append((h_next, flux_next))
        states = new_states
        out_serial.append(_decode_step(model, z_t))
    out_serial = torch.stack(out_serial, dim=1)

    diff_1 = (out_parallel - out_serial).abs().max().item()

    t_ctx, t_pred = 8, 4
    x_context = torch.randn(bsz, t_ctx, channels, height, width, device=device, dtype=torch.float64)
    dt_context_step = torch.ones(bsz, t_ctx - 1, device=device, dtype=torch.float64)
    dt_future_step = torch.full((bsz, t_pred), 0.5, device=device, dtype=torch.float64)

    with torch.no_grad():
        out_optimized = model.forward_rollout(x_context, dt_context_step, dt_future_step)
        out_full_serial = full_serial_inference(model, x_context, dt_context_step, dt_future_step)

    diff_2 = (out_optimized - out_full_serial).abs().max().item()
    passed = diff_1 < 1e-10 and diff_2 < 1e-10
    return passed, diff_1, diff_2


if __name__ == "__main__":
    ok, d1, d2 = check_precision_robustness()
    raise SystemExit(0 if ok else 1)
