import torch

from Check.utils import build_check_model, make_input, max_diff


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    model = build_check_model(device, dtype=dtype, seed=5).eval()
    x = make_input(2, 3, 4, 16, 16, device, dtype)
    dt = torch.zeros(2, 3, device=device, dtype=dtype)
    with torch.no_grad():
        out = model(x, dt)
        roll = model.forward_rollout(x[:, :1], dt[:, :1], [torch.zeros(2, device=device, dtype=dtype)])
    err = max(max_diff(out, x), max_diff(roll[:, 0], x[:, 0]))
    tol = 1e-7 if dtype == torch.float32 else 1e-12
    return ("PASS" if err <= tol else "FAIL"), err, "zero dt maps to identity"


if __name__ == "__main__":
    print(run())
