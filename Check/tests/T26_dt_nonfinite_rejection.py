import sys

import torch

from Model.UniPhy.ModelUniPhy import UniPhyModel
from Check.utils import write_result

TEST_ID = "T26"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=8,
        expand=2,
        depth=1,
        patch_size=(7, 15),
        img_height=721,
        img_width=1440,
        dt_ref=6.0,
        init_noise_scale=1e-4,
    ).to(device)
    cases = [
        torch.tensor([[float("nan")]], device=device),
        torch.tensor([[float("inf")]], device=device),
        torch.tensor([[-float("inf")]], device=device),
    ]
    passed = 0
    for dt in cases:
        try:
            model._validate_dt(dt)
        except ValueError:
            passed += 1
    status = "PASS" if passed == len(cases) else "FAIL"
    max_error = 0.0 if status == "PASS" else 1.0
    detail = f"nonfinite_cases_passed={passed}/{len(cases)}"
    return status, max_error, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
