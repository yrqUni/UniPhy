import torch
import torch.nn as nn
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from PScan import pscan


def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def cmp(name, t1, t2, atol=1e-5):
    diff = (t1 - t2).abs().max().item()
    ok = diff < atol
    print(f"{name}: diff={diff:.2e} [{'PASS' if ok else 'FAIL'}]")
    return ok, diff


def debug_block_state_propagation():
    sep("Block State Propagation Analysis")
    torch.manual_seed(42)

    B, T, D, H, W = 1, 4, 32, 8, 8

    block = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")
    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")

    print("\n--- Parallel Mode Internal States ---")
    with torch.no_grad():
        x_spatial = block._spatial_process(x)
        x_perm = x_spatial.permute(0, 1, 3, 4, 2)
        x_eigen_par = block.prop.basis.encode(x_perm)
        x_mean_par = x_eigen_par.mean(dim=(2, 3))

        flux_list_par = []
        source_list_par = []
        gate_list_par = []
        current_flux = flux_init.clone()

        for t in range(T):
            new_flux, source, gate = block.prop.flux_tracker.forward_step(
                current_flux, x_mean_par[:, t]
            )
            flux_list_par.append(new_flux.clone())
            source_list_par.append(source.clone())
            gate_list_par.append(gate.clone())
            current_flux = new_flux

        flux_seq_par = torch.stack(flux_list_par, dim=1)
        source_seq_par = torch.stack(source_list_par, dim=1)
        gate_seq_par = torch.stack(gate_list_par, dim=1)

        dt_exp = dt.unsqueeze(0).expand(B, T)
        op_decay_par, op_forcing_par = block.prop.get_transition_operators(dt_exp)

        print(f"x_eigen_par shape: {x_eigen_par.shape}")
        print(f"flux_seq_par shape: {flux_seq_par.shape}")
        print(f"op_decay_par shape: {op_decay_par.shape}")

    print("\n--- Serial Mode Internal States ---")
    with torch.no_grad():
        flux_list_ser = []
        source_list_ser = []
        gate_list_ser = []
        h_list_ser = []
        x_eigen_list_ser = []

        h_prev = h_init.clone()
        flux_prev = flux_init.clone()

        for t in range(T):
            x_t = x[:, t]

            x_real = torch.cat([x_t.real, x_t.imag], dim=1)
            x_norm = block.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_spatial_t = block.spatial_cliff(x_norm)
            x_s_re, x_s_im = torch.chunk(x_spatial_t, 2, dim=1)
            x_t = x_t + torch.complex(x_s_re, x_s_im)

            x_perm_t = x_t.permute(0, 2, 3, 1)
            x_eigen_t = block.prop.basis.encode(x_perm_t)
            x_mean_t = x_eigen_t.mean(dim=(1, 2))

            x_eigen_list_ser.append(x_eigen_t.clone())

            flux_next, source, gate = block.prop.flux_tracker.forward_step(flux_prev, x_mean_t)

            flux_list_ser.append(flux_next.clone())
            source_list_ser.append(source.clone())
            gate_list_ser.append(gate.clone())

            source_exp = source.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
            gate_exp = gate.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)

            forcing = x_eigen_t * gate_exp + source_exp * (1 - gate_exp)

            op_decay_t, op_forcing_t = block.prop.get_transition_operators(dt[t])

            h_prev_reshaped = h_prev.reshape(B, H, W, D)
            h_next = h_prev_reshaped * op_decay_t + forcing * op_forcing_t

            h_list_ser.append(h_next.clone())

            h_prev = h_next.reshape(B * H * W, 1, D)
            flux_prev = flux_next

        flux_seq_ser = torch.stack(flux_list_ser, dim=1)
        source_seq_ser = torch.stack(source_list_ser, dim=1)
        gate_seq_ser = torch.stack(gate_list_ser, dim=1)

    print("\n--- Compare Intermediate States ---")

    print("\nFlux comparison per timestep:")
    for t in range(T):
        cmp(f"  flux[{t}]", flux_list_par[t], flux_list_ser[t])

    print("\nSource comparison per timestep:")
    for t in range(T):
        cmp(f"  source[{t}]", source_list_par[t], source_list_ser[t])

    print("\nGate comparison per timestep:")
    for t in range(T):
        cmp(f"  gate[{t}]", gate_list_par[t], gate_list_ser[t])

    print("\nx_eigen comparison per timestep:")
    for t in range(T):
        cmp(f"  x_eigen[{t}]", x_eigen_par[:, t], x_eigen_list_ser[t])

    print("\n--- Analyze PScan vs Serial h_next ---")
    with torch.no_grad():
        source_exp_par = source_seq_par.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        gate_exp_par = gate_seq_par.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        forcing_par = x_eigen_par * gate_exp_par + source_exp_par * (1 - gate_exp_par)

        op_decay_exp = op_decay_par.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing_exp = op_forcing_par.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        u_t_par = forcing_par * op_forcing_exp

        print(f"\nu_t_par shape: {u_t_par.shape}")
        print(f"op_decay_exp shape: {op_decay_exp.shape}")

        A = op_decay_exp.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X = u_t_par.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)

        print(f"A shape for pscan: {A.shape}")
        print(f"X shape for pscan: {X.shape}")

        Y_pscan = pscan(A, X)
        u_out_par = Y_pscan.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        print(f"\nu_out_par (from PScan) shape: {u_out_par.shape}")

    print("\nCompare PScan output vs Serial h_next:")
    for t in range(T):
        u_par_t = u_out_par[:, t]
        h_ser_t = h_list_ser[t]
        cmp(f"  h[{t}]", u_par_t, h_ser_t)

    print("\n--- Analyze PScan Behavior ---")
    print("\nManual loop to verify PScan semantics:")
    with torch.no_grad():
        Y_manual = torch.zeros_like(X)
        acc = torch.zeros(B * H * W, D, 1, dtype=X.dtype, device=X.device)

        for t in range(T):
            acc = acc * A[:, t] + X[:, t]
            Y_manual[:, t] = acc

        u_out_manual = Y_manual.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

    print("\nCompare PScan vs Manual loop:")
    cmp("PScan vs Manual", u_out_par, u_out_manual)

    print("\nCompare Manual loop vs Serial h_next:")
    for t in range(T):
        u_manual_t = u_out_manual[:, t]
        h_ser_t = h_list_ser[t]
        _, diff = cmp(f"  manual[{t}] vs serial[{t}]", u_manual_t, h_ser_t)

        if diff > 1e-4:
            print(f"    Investigating t={t}...")
            print(f"    u_manual_t mean: {u_manual_t.real.mean():.6f}")
            print(f"    h_ser_t mean: {h_ser_t.real.mean():.6f}")

            if t == 0:
                expected = X[:, 0].reshape(B, H, W, D)
                print(f"    X[0] mean: {expected.real.mean():.6f}")

                forcing_ser_0 = x_eigen_list_ser[0] * gate_list_ser[0].unsqueeze(1).unsqueeze(2).expand(B, H, W, D) + \
                                source_list_ser[0].unsqueeze(1).unsqueeze(2).expand(B, H, W, D) * \
                                (1 - gate_list_ser[0].unsqueeze(1).unsqueeze(2).expand(B, H, W, D))
                op_decay_0, op_forcing_0 = block.prop.get_transition_operators(dt[0])
                expected_h0 = forcing_ser_0 * op_forcing_0
                print(f"    expected h[0] from serial: {expected_h0.real.mean():.6f}")


def debug_model_state_flow():
    sep("Model State Flow Analysis")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33, sde_mode="ode"
    ).cuda().eval()

    x = torch.randn(1, 4, 2, 33, 33, device="cuda") * 0.1
    dt = torch.ones(4, device="cuda")

    B, T = 1, 4
    H_p, W_p, D = model.h_patches, model.w_patches, model.embed_dim

    print(f"\nModel config: h_patches={H_p}, w_patches={W_p}, embed_dim={D}, depth={model.depth}")

    with torch.no_grad():
        z = model.encoder(x)
        dtype = z.dtype

        print(f"\nEncoder output shape: {z.shape}")

        states_par = model._init_states(B, z.device, dtype)
        z_par = z.clone()

        all_h_par = []
        all_flux_par = []

        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states_par[i]
            z_par, h_next, flux_next = block(z_par, h_prev, dt, flux_prev)
            states_par[i] = (h_next, flux_next)
            all_h_par.append(h_next.clone())
            all_flux_par.append(flux_next.clone())
            print(f"Block {i} parallel: z shape={z_par.shape}, h shape={h_next.shape}")

        out_par = model.decoder(z_par)

    with torch.no_grad():
        z_ser = model.encoder(x[:, 0:1]).squeeze(1)

        print(f"\nSerial initial z shape: {z_ser.shape}")

        states_ser = model._init_states(B, z_ser.device, dtype)

        all_z_ser = []
        all_h_ser = []
        all_flux_ser = []

        for t in range(T):
            dt_t = dt[t]
            new_states = []

            z_t = z_ser if t == 0 else all_z_ser[-1]

            for i, block in enumerate(model.blocks):
                h_prev, flux_prev = states_ser[i]
                z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt_t, flux_prev)
                new_states.append((h_next, flux_next))

            states_ser = new_states
            all_z_ser.append(z_t.clone())
            all_h_ser.append([s[0].clone() for s in states_ser])
            all_flux_ser.append([s[1].clone() for s in states_ser])

            print(f"t={t}: z shape={z_t.shape}")

    print("\n--- Compare States at Each Timestep ---")

    z_par_per_t = z_par

    for t in range(T):
        print(f"\nTimestep {t}:")

        z_par_t = z_par[:, t]
        z_ser_t = all_z_ser[t]
        cmp(f"  z[{t}]", z_par_t, z_ser_t)

        for i in range(model.depth):
            if t == T - 1:
                h_par_i = all_h_par[i]
                h_ser_i = all_h_ser[t][i]
                cmp(f"  block{i} h[{t}]", h_par_i, h_ser_i)


def debug_encoder_decoder():
    sep("Encoder/Decoder Consistency")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33, sde_mode="ode"
    ).cuda().eval()

    x_5d = torch.randn(1, 4, 2, 33, 33, device="cuda")
    x_4d = x_5d[:, 0]

    with torch.no_grad():
        z_5d = model.encoder(x_5d)
        z_4d = model.encoder(x_4d.unsqueeze(1)).squeeze(1)

        print(f"z_5d shape: {z_5d.shape}")
        print(f"z_4d shape: {z_4d.shape}")

        cmp("Encoder z[0] 5D vs 4D", z_5d[:, 0], z_4d)

        out_5d = model.decoder(z_5d)
        out_4d = model.decoder(z_4d.unsqueeze(1)).squeeze(1)

        print(f"\nout_5d shape: {out_5d.shape}")
        print(f"out_4d shape: {out_4d.shape}")

        cmp("Decoder out[0] 5D vs 4D", out_5d[:, 0], out_4d)


def main():
    print("=" * 70)
    print("  Detailed State Propagation Debug")
    print("=" * 70)

    debug_encoder_decoder()
    debug_block_state_propagation()
    debug_model_state_flow()

    print("\n" + "=" * 70)
    print("  Debug Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    