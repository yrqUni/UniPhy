import os, sys, math, argparse
from dataclasses import dataclass
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../Model/ConvLRU"))
try:
    from ModelConvLRU import *
except Exception as e:
    print(f"[ERROR] failed to import from ../Model/ConvLRU: {e}")
    sys.exit(2)

torch.set_default_dtype(torch.float64)

@dataclass
class CheckItem:
    name: str
    ok: bool
    msg: str = ""
    skip: bool = False

def gauss_legendre(n, device=None, dtype=None):
    if device is None: device = torch.device('cpu')
    if dtype  is None: dtype  = torch.get_default_dtype()
    i = torch.arange(1, n, device=device, dtype=dtype)
    b = i / torch.sqrt(4*i*i - 1)
    J = torch.zeros((n,n), device=device, dtype=dtype)
    J[range(n-1), range(1,n)] = b
    J[range(1,n), range(n-1)] = b
    evals, evecs = torch.linalg.eigh(J)
    x = evals
    w = 2 * (evecs[0, :] ** 2)
    return x, w

def periodic_trap_weights(Wphi, device=None, dtype=None):
    if device is None: device = torch.device('cpu')
    if dtype  is None: dtype  = torch.get_default_dtype()
    dphi = 2*math.pi / Wphi
    return torch.full((Wphi,), dphi, device=device, dtype=dtype)

def idx_lm(l, m):
    return l*l + (m + l)

def has_attrs(obj, *names): 
    return all(hasattr(obj, n) for n in names)

def build_P_table_from_x(x, Lmax, fact_ratio_fn, double_fact_fn):
    device=x.device; dtype=x.dtype
    P = [[None]*(l+1) for l in range(Lmax)]
    P[0][0] = torch.ones_like(x)
    if Lmax>=2: P[1][0]=x
    for l in range(2, Lmax):
        l_f = torch.tensor(l, device=device, dtype=dtype)
        P[l][0] = ((2*l_f-1)*x*P[l-1][0] - (l_f-1)*P[l-2][0]) / l_f
    for m in range(1, Lmax):
        m_f = torch.tensor(m, device=device, dtype=dtype)
        if double_fact_fn is None:
            val = torch.exp(torch.lgamma(torch.tensor(2*m+1.,device=device,dtype=dtype))
                            - (m*math.log(2.0))
                            - torch.lgamma(torch.tensor(m+1.,device=device,dtype=dtype)))
        else:
            val = double_fact_fn(2*m-1, dtype, device)
        P_mm = ((-1)**m) * val * (1 - x*x).pow(m_f/2)
        P[m][m] = P_mm
        if m+1 < Lmax: P[m+1][m] = (2*m_f+1)*x*P_mm
        for l in range(m+2, Lmax):
            l_f = torch.tensor(l, device=device, dtype=dtype)
            P[l][m] = ((2*l_f-1)*x*P[l-1][m] - (l_f+m_f-1)*P[l-2][m]) / (l_f-m_f)
    return P

def check_spectral_prior(spectral_cls, device="cpu", seed=0):
    torch.manual_seed(seed)
    items = []
    C,S,W,R = 3, 12, 10, 4
    B,L = 2, 3
    h = torch.randn(B, L, C, S, W, device=device)
    ok=True; msgs=[]
    for mode in ["linear","exp"]:
        try:
            sp = spectral_cls(C,S,W,rank=R,gain_init=0.0,mode=mode).to(device)
            diff = (sp(h)-h).abs().max().item()
            ok &= diff < 1e-12
            msgs.append(f"{mode}: max|Δ|={diff:.3e}")
        except Exception as e:
            ok=False; msgs.append(f"{mode}: EXC={e}")
    items.append(CheckItem("SpectralPrior2D identity (gain=0)", ok, "; ".join(msgs)))
    try:
        sp = spectral_cls(C,S,W,rank=R,gain_init=0.0,mode="linear").to(device)
        if not has_attrs(sp, "A","B"):
            items.append(CheckItem("SpectralPrior2D F outer-sum == einsum", False, "missing A/B", skip=True))
        else:
            with torch.no_grad():
                A = sp.A; Bp = sp.B
                F_ein = torch.einsum("cri,crj->cij", A, Bp)
                F_sum = torch.stack([sum(A[c,r].unsqueeze(1)@Bp[c,r].unsqueeze(0) for r in range(R)) for c in range(C)], dim=0)
            diff = (F_ein - F_sum).abs().max().item()
            items.append(CheckItem("SpectralPrior2D F outer-sum == einsum", diff<1e-12, f"max|ΔF|={diff:.3e}"))
    except Exception as e:
        items.append(CheckItem("SpectralPrior2D F outer-sum == einsum", False, f"EXC={e}"))
    try:
        sp = spectral_cls(C,S,W,rank=R,gain_init=0.2,mode="linear").to(device)
        if not has_attrs(sp, "A","B","gain"):
            items.append(CheckItem("SpectralPrior2D broadcasting check", False, "missing A/B/gain", skip=True))
        else:
            with torch.no_grad():
                F = torch.einsum("cri,crj->cij", sp.A, sp.B)
                G = 1.0 + sp.gain.view(C,1,1) * F
            out = sp(h)
            manual = h * G.view(1,1,C,S,W)
            diff = (out - manual).abs().max().item()
            items.append(CheckItem("SpectralPrior2D broadcasting check", diff<1e-12, f"max|Δ|={diff:.3e}"))
    except Exception as e:
        items.append(CheckItem("SpectralPrior2D broadcasting check", False, f"EXC={e}"))
    return items

def check_sh_suite(sh_cls, device="cpu", seed=0, Lmax=8, Ntheta=120, Nphi=192):
    torch.manual_seed(seed)
    items = []
    has_basis = hasattr(sh_cls, "_real_sph_harm_basis")
    has_fact  = hasattr(sh_cls, "_fact_ratio")
    has_dblf  = hasattr(sh_cls, "_double_factorial")
    try:
        if has_basis:
            x, wx = gauss_legendre(Ntheta, device=device)
            theta = torch.arccos(x).unsqueeze(1).repeat(1, Nphi)
            phi = torch.linspace(-math.pi, math.pi, steps=Nphi+1, device=device)[:-1].unsqueeze(0).repeat(Ntheta,1)
            Y = sh_cls._real_sph_harm_basis(theta, phi, Lmax)
            K = Lmax*Lmax
            W = wx.view(1,-1,1) * periodic_trap_weights(Nphi, device=device).view(1,1,-1)
            G = ((Y*W).reshape(K,-1)) @ (Y.reshape(K,-1).t())
            I = torch.eye(K, dtype=Y.dtype, device=Y.device)
            err_max = (G-I).abs().max().item(); err_fro = torch.linalg.norm(G-I).item()
            items.append(CheckItem("Real SH orthonormality (GL×periodic φ)", err_max<5e-10 and err_fro<5e-9,
                                   f"max|G-I|={err_max:.3e}, ||G-I||_F={err_fro:.3e}"))
        else:
            shp = sh_cls(channels=1, H=121, W=181, Lmax=Lmax, rank=2, gain_init=0.0).to(device)
            Y = shp.Y_real
            H, Ww = Y.shape[-2:]
            lat = torch.linspace(math.pi/2, -math.pi/2, steps=H, device=device)
            lon = torch.linspace(-math.pi, math.pi, steps=Ww, device=device)
            theta = (math.pi/2 - lat).unsqueeze(1).repeat(1, Ww)
            dtheta = (theta.max()-theta.min())/(H-1); dphi = (lon.max()-lon.min())/(Ww-1)
            wt = torch.ones(H, device=device); wt[0]=0.5; wt[-1]=0.5
            wp = torch.ones(Ww, device=device); wp[0]=0.5; wp[-1]=0.5
            Wgt = torch.sin(theta)*wt.view(H,1)*wp.view(1,Ww)*dtheta*dphi
            K = Lmax*Lmax
            G = ((Y*Wgt).reshape(K,-1)) @ (Y.reshape(K,-1).t())
            I = torch.eye(K, dtype=Y.dtype, device=Y.device)
            err_max = (G-I).abs().max().item(); err_fro = torch.linalg.norm(G-I).item()
            items.append(CheckItem("Real SH orthonormality (lat/lon trapezoid)", err_max<2e-3 and err_fro<5e-2,
                                   f"max|G-I|={err_max:.3e}, ||G-I||_F={err_fro:.3e}"))
    except Exception as e:
        items.append(CheckItem("Real SH orthonormality", False, f"EXC={e}"))
    try:
        x, _ = gauss_legendre(max(Ntheta//1, 30), device=device)
        theta = torch.arccos(x).unsqueeze(1).repeat(1, Nphi)
        phi = torch.linspace(-math.pi, math.pi, steps=Nphi+1, device=device)[:-1].unsqueeze(0).repeat(theta.shape[0],1)
        if has_basis:
            Y = sh_cls._real_sph_harm_basis(theta, phi, Lmax)
        else:
            items.append(CheckItem("Real↔Complex consistency", False, "missing _real_sph_harm_basis", skip=True))
            Y = None
        def fact_ratio(l, m):
            if has_fact:
                return sh_cls._fact_ratio(l, m, x.dtype, x.device)
            l_t = torch.tensor(l, dtype=x.dtype, device=x.device)
            m_t = torch.tensor(m, dtype=x.dtype, device=x.device)
            return torch.exp(torch.lgamma(l_t - m_t + 1) - torch.lgamma(l_t + m_t + 1))
        def N_lm(l, m):
            return torch.sqrt((2*l+1)/(4*torch.tensor(math.pi, dtype=x.dtype, device=x.device)) * fact_ratio(l, m))
        P = build_P_table_from_x(x.view(-1,1), Lmax,
                                 fact_ratio_fn=sh_cls._fact_ratio if has_fact else None,
                                 double_fact_fn=sh_cls._double_factorial if has_dblf else None)
        ok=True; max_err=0.0
        if Y is not None:
            for l in range(1, Lmax):
                for m in range(1, l+1):
                    Yr_c = Y[idx_lm(l,+m)]
                    Yr_s = Y[idx_lm(l,-m)]
                    Ycplx = (Yr_c + 1j*Yr_s)/math.sqrt(2.0)
                    ref = N_lm(l,m) * P[l][m] * (torch.cos(m*phi) + 1j*torch.sin(m*phi))
                    err = (Ycplx - ref).abs().max().item()
                    ok &= err < 1e-12
                    max_err = max(max_err, err)
            items.append(CheckItem("Real↔Complex SH consistency", ok, f"max|Δ|={max_err:.3e}"))
    except Exception as e:
        items.append(CheckItem("Real↔Complex SH consistency", False, f"EXC={e}"))
    try:
        if has_basis:
            x, wx = gauss_legendre(Ntheta, device=device)
            theta = torch.arccos(x).unsqueeze(1).repeat(1, Nphi)
            phi = torch.linspace(-math.pi, math.pi, steps=Nphi+1, device=device)[:-1].unsqueeze(0).repeat(Ntheta,1)
            Y = sh_cls._real_sph_harm_basis(theta, phi, Lmax)
            K = Lmax*Lmax
            W = wx.view(1,-1,1) * periodic_trap_weights(Nphi, device=device).view(1,1,-1)
            a = torch.randn(K, device=device)
            f = (a.view(K,1,1)*Y).sum(0)
            a_rec = ((Y*W).reshape(K,-1) @ f.reshape(-1))
            err = (a_rec - a).abs().max().item()
            items.append(CheckItem("Roundtrip projection (GL)", err<5e-10, f"max|a_rec-a|={err:.3e}"))
        else:
            items.append(CheckItem("Roundtrip projection", False, "missing _real_sph_harm_basis", skip=True))
    except Exception as e:
        items.append(CheckItem("Roundtrip projection (GL)", False, f"EXC={e}"))
    try:
        if has_basis:
            x, wx = gauss_legendre(Ntheta, device=device)
            theta = torch.arccos(x).unsqueeze(1).repeat(1, Nphi)
            phi = torch.linspace(-math.pi, math.pi, steps=Nphi+1, device=device)[:-1].unsqueeze(0).repeat(Ntheta,1)
            Y = sh_cls._real_sph_harm_basis(theta, phi, Lmax)
            K = Lmax*Lmax
            W = wx.view(1,-1,1) * periodic_trap_weights(Nphi, device=device).view(1,1,-1)
            ints = (Y*W).reshape(K,-1).sum(-1)
            vals = [abs(ints[idx_lm(l,m)].item()) for l in range(1,Lmax) for m in range(-l,l+1)]
            mx = max(vals) if vals else 0.0
            items.append(CheckItem("Mean-zero for l>0", mx<5e-12, f"max|∫Y|={mx:.3e}"))
        else:
            items.append(CheckItem("Mean-zero for l>0", False, "missing _real_sph_harm_basis", skip=True))
    except Exception as e:
        items.append(CheckItem("Mean-zero for l>0", False, f"EXC={e}"))
    try:
        C,H,Wv,R = 3, 61, 91, 3
        shp = sh_cls(channels=C, H=H, W=Wv, Lmax=Lmax, rank=R, gain_init=0.7)
        shp = shp.to(device)
        x_in = torch.randn(2,4,C,H,Wv, device=device, dtype=torch.float64)
        if not has_attrs(shp, "W1","W2","gain","Y_real"):
            items.append(CheckItem("SH prior forward consistency", False, "missing W1/W2/gain/Y_real", skip=True))
        else:
            Y = shp.Y_real.to(dtype=x_in.dtype, device=device)
            coeff = shp.W1 @ shp.W2
            bias = torch.einsum('ck,khw->chw', coeff, Y)
            bias = (shp.gain.view(C,1,1)*bias).view(1,1,C,H,Wv)
            out = shp(x_in); diff = (out - (x_in + bias)).abs().max().item()
            items.append(CheckItem("SH prior forward consistency", diff<1e-12, f"max|Δ|={diff:.3e}"))
            shp2 = sh_cls(channels=C, H=H, W=Wv, Lmax=Lmax, rank=1, gain_init=1.0).to(device)
            if has_attrs(shp2, "W1","W2","gain","Y_real"):
                with torch.no_grad():
                    shp2.W1.zero_(); shp2.W2.zero_()
                    c = 1; l,m = 3,-2
                    k = idx_lm(l,m)
                    shp2.W1[c,0] = 1.0; shp2.W2[0,k] = 1.0
                out2 = shp2(torch.zeros(1,1,C,H,Wv, device=device, dtype=torch.float64))
                diff2 = (out2[0,0,c] - shp2.Y_real[k]).abs().max().item()
                items.append(CheckItem("SH prior one-hot reproduces basis", diff2<1e-12, f"max|bias-Y_lm|={diff2:.3e}"))
            else:
                items.append(CheckItem("SH prior one-hot reproduces basis", False, "missing W1/W2/gain/Y_real", skip=True))
    except Exception as e:
        items.append(CheckItem("SH prior forward/one-hot", False, f"EXC={e}"))
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lmax", type=int, default=8)
    parser.add_argument("--ntheta", type=int, default=120)
    parser.add_argument("--nphi", type=int, default=192)
    args = parser.parse_args()
    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    if "SpectralPrior2D" not in globals() or not isinstance(globals()["SpectralPrior2D"], type):
        print("[ERROR] class SpectralPrior2D not found in ModelConvLRU")
        sys.exit(2)
    if "SphericalHarmonicsPrior" not in globals() or not isinstance(globals()["SphericalHarmonicsPrior"], type):
        print("[ERROR] class SphericalHarmonicsPrior not found in ModelConvLRU")
        sys.exit(2)
    spectral_cls = globals()["SpectralPrior2D"]
    sh_cls = globals()["SphericalHarmonicsPrior"]
    results = []
    results += check_spectral_prior(spectral_cls, device=device, seed=args.seed)
    results += check_sh_suite(sh_cls, device=device, seed=args.seed, Lmax=args.lmax,
                              Ntheta=args.ntheta, Nphi=args.nphi)
    print("="*80)
    print("Results (PASS/FAIL/SKIP):")
    n_pass = n_fail = n_skip = 0
    for it in results:
        tag = "SKIP" if it.skip else ("PASS" if it.ok else "FAIL")
        if it.skip: n_skip += 1
        elif it.ok: n_pass += 1
        else: n_fail += 1
        extra = f": {it.msg}" if it.msg else ""
        print(f"[{tag}] {it.name}{extra}")
    print("-"*80)
    print(f"Passed {n_pass}, Failed {n_fail}, Skipped {n_skip}")
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()
