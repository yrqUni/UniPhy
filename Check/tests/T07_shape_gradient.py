import torch

from Check.utils import build_check_model, make_dt, make_input


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model = build_check_model(device, dtype=dtype, seed=7).train()
    x = make_input(1, 2, 4, 16, 16, device, dtype)
    dt = make_dt(1, 2, device, dtype)
    out = model(x, dt)
    loss = out.square().mean()
    loss.backward()
    bad = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            bad += 1
    expected_shape = tuple(x.shape)
    ok = tuple(out.shape) == expected_shape and bad == 0
    return ("PASS" if ok else "FAIL"), float(bad), f"shape={tuple(out.shape)}"


if __name__ == "__main__":
    print(run())
