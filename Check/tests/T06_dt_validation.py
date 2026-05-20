import torch

from Check.utils import build_check_model, make_input


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    model = build_check_model(device, dtype=dtype, seed=6).eval()
    x = make_input(1, 2, 4, 16, 16, device, dtype)
    failures = 0
    for dt in [torch.tensor([[-1.0, 1.0]], device=device, dtype=dtype), torch.tensor([[float("nan"), 1.0]], device=device, dtype=dtype)]:
        try:
            model(x, dt)
            failures += 1
        except ValueError:
            pass
    return ("PASS" if failures == 0 else "FAIL"), float(failures), "negative and nonfinite dt rejected"


if __name__ == "__main__":
    print(run())
