import torch
import torch.nn as nn
from GrandUnified import CliffordConv2d, StochasticHamiltonianSSM, StreamFunctionMixing, fused_curl_2d, FusedHamiltonian

def check_clifford_conv():
    print("Checking CliffordConv2d...")
    B, C, H, W = 2, 16, 64, 64
    x = torch.randn(B, C, H, W).cuda()
    model = CliffordConv2d(dim=C, kernel_size=3, padding=1).cuda()
    out = model(x)
    assert out.shape == x.shape
    print("  Pass: Output shape matches input.")

def check_stochastic_hamiltonian_ssm():
    print("Checking StochasticHamiltonianSSM...")
    B, C, H, W = 1, 16, 64, 64
    z = torch.randn(B, C, H, W).cuda()
    dt = torch.full((B,), 0.1).cuda()
    model = StochasticHamiltonianSSM(hidden_dim=C, h=H, w=W).cuda()
    out = model(z, dt)
    assert out.shape == z.shape
    print("  Pass: Forward execution successful.")

def check_stream_function_mixing():
    print("Checking StreamFunctionMixing...")
    B, C, H, W = 2, 16, 64, 64
    z = torch.randn(B, C, H, W).cuda()
    dt = torch.full((B,), 0.1).cuda() 
    model = StreamFunctionMixing(in_ch=C, h=H, w=W).cuda()
    out = model(z, dt)
    assert out.shape == z.shape
    print("  Pass: Flow generation and grid sampling successful.")

def check_fused_ops():
    print("Checking Low-level Triton Ops...")
    B, C, H, W = 2, 4, 64, 64
    psi = torch.randn(B, C, H, W).cuda()
    u, v = fused_curl_2d(psi)
    assert u.shape == psi.shape
    assert v.shape == psi.shape
    print("  Pass: Fused Curl 2D.")

    B_ham = 2 
    C_ham = 16
    H_ham, W_ham = 64, 33
    z_real = torch.randn(B_ham, C_ham, H_ham, W_ham).cuda()
    z_imag = torch.randn(B_ham, C_ham, H_ham, W_ham).cuda()
    h_real = torch.randn(C_ham, H_ham, W_ham).cuda()
    h_imag = torch.randn(C_ham, H_ham, W_ham).cuda()
    dt = torch.full((B_ham,), 0.1).cuda()
    sigma = torch.tensor(0.01).cuda()

    out_r, out_i = FusedHamiltonian.apply(z_real, z_imag, h_real, h_imag, dt, sigma)
    assert out_r.shape == z_real.shape
    assert out_i.shape == z_imag.shape
    print("  Pass: Fused Hamiltonian Kernel.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        check_clifford_conv()
        check_stochastic_hamiltonian_ssm()
        check_stream_function_mixing()
        check_fused_ops()
        print("\nAll checks passed successfully.")
    else:
        print("CUDA required for Triton kernels. Skipping checks.")

