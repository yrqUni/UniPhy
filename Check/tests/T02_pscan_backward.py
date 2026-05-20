import torch

from Check.utils import max_diff
from Model.UniPhy.PScan import pscan_torch_tree


def run_case(device):
    torch.manual_seed(2)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    a1 = torch.sigmoid(torch.randn(2, 8, 3, 4, device=device, dtype=dtype)).requires_grad_(True)
    x1 = torch.randn(2, 8, 3, 4, 2, device=device, dtype=dtype, requires_grad=True)
    a2 = a1.detach().clone().requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = pscan_torch_tree(a1, x1).square().mean()
    y2 = []
    state = torch.zeros_like(x2[:, 0])
    for idx in range(x2.shape[1]):
        state = a2[:, idx].unsqueeze(-1) * state + x2[:, idx]
        y2.append(state)
    y2 = torch.stack(y2, dim=1).square().mean()
    y1.backward()
    y2.backward()
    return max(max_diff(a1.grad, a2.grad), max_diff(x1.grad, x2.grad))


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    err = run_case(device)
    tol = 1e-5 if device.type == "cuda" else 1e-10
    return ("PASS" if err <= tol else "FAIL"), err, f"device={device}"


if __name__ == "__main__":
    print(run())
