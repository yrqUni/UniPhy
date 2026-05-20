import torch

from Check.utils import max_diff, serial_diag_scan, serial_mat_scan
from Model.UniPhy.PScan import pscan, pscan_torch_tree


def run():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    a = torch.sigmoid(torch.randn(2, 9, 3, 4, device=device, dtype=dtype))
    x = torch.randn(2, 9, 3, 4, 2, device=device, dtype=dtype)
    ref = serial_diag_scan(a, x)
    err_tree = max_diff(pscan_torch_tree(a, x), ref)
    err_auto = max_diff(pscan(a, x), ref)
    mat = torch.randn(2, 7, 2, 3, 3, device=device, dtype=dtype) * 0.05
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
    mat = mat + eye
    xm = torch.randn(2, 7, 2, 3, 2, device=device, dtype=dtype)
    refm = serial_mat_scan(mat, xm)
    err_mat = max_diff(pscan_torch_tree(mat, xm), refm)
    err = max(err_tree, err_auto, err_mat)
    tol = 5e-6 if dtype == torch.float32 else 1e-10
    status = "PASS" if err <= tol else "FAIL"
    return status, err, f"diag_tree={err_tree:.3e} diag_auto={err_auto:.3e} mat_tree={err_mat:.3e}"


if __name__ == "__main__":
    status, max_error, detail = run()
    print(status, max_error, detail)
