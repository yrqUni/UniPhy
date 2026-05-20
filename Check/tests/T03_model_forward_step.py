import torch

from Check.utils import build_check_model, make_dt, make_input, manual_forward, max_diff


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    model = build_check_model(device, dtype=dtype, seed=3).eval()
    x = make_input(2, 4, 4, 16, 16, device, dtype)
    dt = make_dt(2, 4, device, dtype)
    with torch.no_grad():
        ref = manual_forward(model, x, dt)
        out = model(x, dt)
    err = max_diff(out, ref)
    tol = 5e-6 if dtype == torch.float32 else 1e-10
    return ("PASS" if err <= tol else "FAIL"), err, f"device={device}"


if __name__ == "__main__":
    print(run())
