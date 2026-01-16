import torch
import torch.nn as nn
from UniPhyOps import HamiltonianPropagator, UniPhyBlock, UniPhyNet

def report(test_name, error, threshold=1e-4):
    passed = error < threshold
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"[{test_name}] Max Error: {error:.2e} -> {color}{status}{reset}")
    if not passed:
        raise ValueError(f"{test_name} Failed!")

def check_energy_conservation():
    print("\n--- Checking Hamiltonian Conservation (Unitary Evolution) ---")
    B, T, D = 2, 100, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HamiltonianPropagator(D, conserve_energy=True).to(device)
    
    h_init = torch.randn(B, 1, 1, D, dtype=torch.complex64, device=device)
    h_init = h_init / h_init.abs() 
    
    dt = torch.rand(B, T, device=device) * 0.1
    
    h_curr = h_init.clone()
    energies = []
    
    with torch.no_grad():
        for t in range(T):
            dummy_x = torch.zeros(B, 1, 1, D, device=device)
            _, h_next = model.step_serial(dummy_x, dt[:, t], h_curr)
            
            energy = h_next.abs().mean()
            energies.append(energy)
            h_curr = h_next
            
    energies = torch.stack(energies)
    initial_energy = h_init.abs().mean()
    max_divergence = (energies - initial_energy).abs().max()
    
    report("Conservation (Pure Rotation)", max_divergence, threshold=1e-5)

def check_parallel_serial_consistency():
    print("\n--- Checking Parallel vs Serial Equivalence ---")
    B, T, H, W, C = 2, 32, 8, 8, 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    block = UniPhyBlock(C, input_shape=(H, W)).to(device)
    block.eval()
    
    x_seq = torch.randn(B, T, H, W, C, device=device)
    dt_seq = torch.rand(B, T, device=device) * 0.1
    
    with torch.no_grad():
        out_parallel, h_final_parallel = block.forward_parallel(x_seq, dt_seq)
        
    out_serial_list = []
    h_curr = torch.zeros(B, H, W, C, dtype=torch.complex64, device=device)
    
    with torch.no_grad():
        for t in range(T):
            x_step = x_seq[:, t]
            dt_step = dt_seq[:, t]
            
            x_out, h_curr = block.step_serial(x_step, dt_step, h_curr)
            out_serial_list.append(x_out)
            
    out_serial = torch.stack(out_serial_list, dim=1)
    
    diff_out = (out_parallel - out_serial).abs().max()
    report("Output Tensor Consistency", diff_out, threshold=1e-4)
    
    diff_state = (h_final_parallel - h_curr).abs().max()
    report("Hidden State Consistency", diff_state, threshold=1e-4)

def check_chunked_inference():
    print("\n--- Checking Chunked/Prefill Consistency ---")
    B, T, H, W, C = 2, 64, 4, 4, 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_idx = 32
    
    model = UniPhyNet(in_ch=C, dim=C, input_shape=(H, W), num_layers=2).to(device)
    model.eval()
    
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device)
    
    with torch.no_grad():
        out_full, _ = model.forward_parallel(x, dt)
        
    x1, x2 = x[:, :split_idx], x[:, split_idx:]
    dt1, dt2 = dt[:, :split_idx], dt[:, split_idx:]
    
    with torch.no_grad():
        propagator = model.layers[0].propagator
        x_prop = x.permute(0, 1, 3, 4, 2)
        x1_p, x2_p = x_prop[:, :split_idx], x_prop[:, split_idx:]
        
        out_prop_full, _ = propagator.forward_parallel(x_prop, dt)
        
        out_prop_1, h_split = propagator.forward_parallel(x1_p, dt1)
        out_prop_2, _ = propagator.forward_parallel(x2_p, dt2, initial_state=h_split)
        
        out_prop_chunked = torch.cat([out_prop_1, out_prop_2], dim=1)
        
    diff_chunk = (out_prop_full - out_prop_chunked).abs().max()
    report("Chunked State Injection", diff_chunk, threshold=1e-4)

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    
    try:
        check_energy_conservation()
        check_parallel_serial_consistency()
        check_chunked_inference()
        print("\nAll Checks Passed Successfully. UniPhyOps is robust.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")

