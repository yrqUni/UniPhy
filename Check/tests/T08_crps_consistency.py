import torch

from Exp.ERA5.runtime_config import compute_weighted_crps


def reference(pred, target, weights):
    mae = (weights * (pred - target.unsqueeze(0)).abs()).mean()
    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    members = pred.shape[0]
    for i in range(members):
        for j in range(i + 1, members):
            total = total + (weights * (pred[i] - pred[j]).abs()).mean()
    return mae - total / (members * members)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(8)
    pred = torch.randn(3, 2, 4, 5, 6, 7, device=device)
    target = torch.randn(2, 4, 5, 6, 7, device=device)
    weights = torch.rand(1, 1, 1, 6, 1, device=device)
    err = float((compute_weighted_crps(pred, target, weights) - reference(pred, target, weights)).abs().item())
    return ("PASS" if err <= 1e-7 else "FAIL"), err, "weighted CRPS matches pairwise reference"


if __name__ == "__main__":
    print(run())
