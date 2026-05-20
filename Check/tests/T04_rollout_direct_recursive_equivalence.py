import torch

from Check.utils import build_check_model, make_input, max_diff


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    model = build_check_model(device, dtype=dtype, seed=4).eval()
    x = make_input(2, 1, 4, 16, 16, device, dtype)
    dt_context = torch.zeros(2, 1, device=device, dtype=dtype)
    with torch.no_grad():
        direct = model.forward_rollout(x, dt_context, [torch.full((2,), 4.0, device=device, dtype=dtype)])
        recursive = model.forward_rollout(x, dt_context, [torch.ones(2, device=device, dtype=dtype) for _ in range(4)])
    err = max_diff(direct[:, 0], recursive[:, -1])
    tol = 5e-6 if dtype == torch.float32 else 1e-10
    return ("PASS" if err <= tol else "FAIL"), err, "4h direct equals four 1h internal steps"


if __name__ == "__main__":
    print(run())
