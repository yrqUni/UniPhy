import torch
import torch.nn as nn
import torch.nn.functional as F
from UniPhyOps import UniPhyLayer
import traceback

def report(name, val_start, val_end, metric_name):
    print(f"[{name}] {metric_name}: {val_start:.4f} -> {val_end:.4f}")
    if abs(val_end - val_start) > 1e-4:
        print(f"   >>> PROOF: {name} mechanism is ACTIVE.")
    else:
        print(f"   >>> PROOF: {name} mechanism is INACTIVE.")
    print("-" * 50)

def main():
    print("\n=== GeoSplit-Net Design Proof (Orthogonality Check) ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = 64, 64
    C = 32
    try:
        model = UniPhyLayer(emb_ch=C, input_shape=(H, W), rank=16).to(device)
    except Exception as e:
        print(f"Model init failed: {e}")
        traceback.print_exc()
        return

    model.eval()
    dt = torch.ones(1, device=device) * 1.0

    print("Test Case A: Pure Advection (The Conveyor Belt)")
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    blob = torch.exp(-(x**2 + y**2) / 0.05).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor = blob.repeat(1, C, 1, 1)

    with torch.no_grad():
        model.transport_op.net[-1].weight.fill_(0)
        model.transport_op.net[-1].bias.fill_(0)
        model.transport_op.net[-1].bias[0] = 5.0
        model.transport_op.net[-1].bias[1] = 0.0

    with torch.no_grad():
        out_moved = model.transport_op(input_tensor, dt)

    def get_center_of_mass(img):
        img = img[0,0]
        sum_mass = img.sum() + 1e-6
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_y, grid_x = grid_y.to(device), grid_x.to(device)
        cy = (img * grid_y).sum() / sum_mass
        cx = (img * grid_x).sum() / sum_mass
        return cx.item(), cy.item()

    cx_start, cy_start = get_center_of_mass(input_tensor)
    cx_end, cy_end = get_center_of_mass(out_moved)

    report("Transport", cx_start, cx_end, "Center of Mass X")
    if cx_end > cx_start + 1:
        print("   [SUCCESS] Object moved right.")
    else:
        print("   [FAILURE] Object did not move.")

    print("Test Case B: Pure Diffusion (The Ink Drop)")
    with torch.no_grad():
        model.dispersion_op.estimator[-1].weight.fill_(0)
        model.dispersion_op.estimator[-1].bias.fill_(0)
        model.dispersion_op.estimator[-1].bias[:16] = 5.0
        model.dispersion_op.estimator[-1].bias[16:] = 0.0

    with torch.no_grad():
        out_diffused = model.dispersion_op(input_tensor, dt)

    energy_start = (input_tensor ** 2).sum().item()
    energy_end = (out_diffused ** 2).sum().item()

    report("Dispersion", energy_start, energy_end, "Total Energy")
    if energy_end < energy_start * 0.9:
        print("   [SUCCESS] Energy dissipated.")
    else:
        print("   [FAILURE] Energy preserved.")

    print("Test Case C: Geometric Interaction (The Vector Mix)")
    vec_input = torch.zeros_like(input_tensor)
    vec_input[:, 0, :, :] = blob
    with torch.no_grad():
        out_interact = model.interaction_op[0](vec_input)
    target_vector_ch = 8
    energy_output_ch8 = (out_interact[:, target_vector_ch] ** 2).sum().item()
    print(f"[Interaction] Output Energy in Vector Channel: {energy_output_ch8:.4f}")
    if energy_output_ch8 > 1e-5:
        print("   >>> PROOF: Interaction mechanism is ACTIVE.")
    else:
        print("   >>> PROOF: Interaction mechanism is INACTIVE.")

    print("Test Case D: Pressure Projection (Helmholtz)")
    div_input = torch.randn(1, C, H, W, device=device)
    with torch.no_grad():
        u_f = torch.fft.fftn(div_input[:, 0:1], dim=(-2, -1))
        v_f = torch.fft.fftn(div_input[:, 1:2], dim=(-2, -1))
        kx = model.projection_op.kx
        ky = model.projection_op.ky
        div_start = torch.fft.ifftn(1j * kx * u_f + 1j * ky * v_f, dim=(-2, -1)).real.norm().item()
        
        out_projected = model.projection_op(div_input)
        
        u_f_p = torch.fft.fftn(out_projected[:, 0:1], dim=(-2, -1))
        v_f_p = torch.fft.fftn(out_projected[:, 1:2], dim=(-2, -1))
        div_end = torch.fft.ifftn(1j * kx * u_f_p + 1j * ky * v_f_p, dim=(-2, -1)).real.norm().item()

    report("Projection", div_start, div_end, "Divergence Norm")
    if div_end < div_start * 0.1:
        print("   [SUCCESS] Divergence eliminated via Helmholtz Projection.")
    else:
        print("   [FAILURE] Divergence still present.")

    print("\n=== All Proofs Finished ===")

if __name__ == "__main__":
    main()

