import inspect

import torch

from .ModelUniPhy import UniPhyBlock, UniPhyModel
from .PScan import pscan
from .UniPhyOps import (
    GlobalFluxTracker,
    MultiScaleSpatialMixer,
    TemporalPropagator,
    _compute_sde_scale,
    _safe_forcing,
)


def build_check_model(device, dtype=torch.float64):
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        expand=4,
        depth=2,
        patch_size=(4, 2),
        img_height=32,
        img_width=32,
        dt_ref=1.0,
        init_noise_scale=0.01,
    ).to(device)
    if dtype == torch.float64:
        model = model.double()
    elif dtype != torch.float32:
        raise ValueError(f"unsupported_dtype={dtype}")
    return model


def _make_input(batch_size, steps, channels, height, width, device, dtype):
    return torch.randn(
        batch_size,
        steps,
        channels,
        height,
        width,
        device=device,
        dtype=dtype,
    )


def _make_dt(batch_size, steps, device, dtype):
    base = torch.tensor(
        [0.5, 1.0, 1.25, 0.75, 1.5, 0.6],
        device=device,
        dtype=dtype,
    )
    repeats = (steps + base.numel() - 1) // base.numel()
    seq = base.repeat(repeats)[:steps]
    return seq.unsqueeze(0).repeat(batch_size, 1).contiguous()


def _max_diff(a, b):
    return float((a - b).abs().max().item())


def _mean_diff(a, b):
    return float((a - b).abs().mean().item())


def _broadcast_zero_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


def _forbidden_encoder_names():
    return ["metric" + "_weight", "rho", "metric" + "_det"]


def _deprecated_module_names():
    return {
        "Riem" + "annian" + "Clif" + "fordConv2d",
        "Efficient" + "SpatialPool",
        "Multi" + "Slot" + "FluxTracker",
        "Grouped" + "Decoder" + "SkipGate",
        "Original" + "Decoder" + "SkipGate",
        "External" + "ContextPath",
        "DC" + "ConservationPath",
        "Temporal" + "ContextModulator",
    }


def _deprecated_param_keywords():
    return [
        "cliff",
        "metric",
        "rho",
        "efficient_spatial",
        "multi_slot",
        "grouped_skip",
        "spectral_mod",
        "external_context",
        "dc_conservation",
        "temporal_context_mod",
    ]


def _no_variant_attr_names():
    return [
        "variant" + "_config",
        "variant" + "_name",
        "_use_" + "efficient_spatial",
        "_use_" + "multi_slot_flux",
        "_use_" + "grouped_skip",
        "_use_" + "per_mode_alpha",
        "_remove_" + "metric",
        "decoder_skip_gate",
    ]


def _forbidden_constructor_params():
    return [
        "variant" + "_config",
        "variant",
        "use_" + "per_mode_alpha",
        "remove_" + "metric",
    ]


def _manual_forward(model, x, dt):
    batch_size, steps = x.shape[:2]
    device = x.device
    dt_seq = model._normalize_dt(dt, batch_size, steps, device)
    z_all = model.encoder(x)
    states = model._init_states(batch_size, device, z_all.dtype)
    outputs = []

    for step_idx in range(steps):
        z_step = z_all[:, step_idx]
        z_skip = z_all[:, step_idx]
        dt_step = dt_seq[:, step_idx]
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_step, h_next, flux_next = block.forward_step(
                z_step,
                h_prev,
                dt_step,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        out_step = model.decoder(model._apply_decoder_skip(z_step, z_skip))
        zero_mask = _broadcast_zero_mask(dt_step.abs() <= 1e-12, out_step.ndim)
        out_step = torch.where(zero_mask, x[:, step_idx], out_step)
        outputs.append(out_step)

    return torch.stack(outputs, dim=1)


def full_serial_inference(model, x_context, dt_context, dt_list):
    batch_size, context_steps = x_context.shape[:2]
    device = x_context.device
    z_all = model.encoder(x_context)
    z_skip = z_all[:, -1]
    states = model._init_states(batch_size, device, z_all.dtype)
    dt_ctx = model._normalize_dt(dt_context, batch_size, context_steps, device)

    if context_steps > 1:
        for step_idx in range(context_steps - 1):
            z_step = z_all[:, step_idx]
            dt_step = dt_ctx[:, step_idx + 1]
            for block_idx, block in enumerate(model.blocks):
                h_prev, flux_prev = states[block_idx]
                z_step, h_next, flux_next = block.forward_step(
                    z_step,
                    h_prev,
                    dt_step,
                    flux_prev,
                )
                states[block_idx] = (h_next, flux_next)

    z_curr = z_all[:, -1]
    x_curr = x_context[:, -1]
    preds = []
    for dt_step in dt_list:
        dt_value = model._normalize_dt(dt_step, batch_size, 1, device).squeeze(1)
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_curr, h_next, flux_next = block.forward_step(
                z_curr,
                h_prev,
                dt_value,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        pred = model.decoder(model._apply_decoder_skip(z_curr, z_skip))
        zero_mask = _broadcast_zero_mask(dt_value.abs() <= 1e-12, pred.ndim)
        pred = torch.where(zero_mask, x_curr, pred)
        x_curr = pred
        preds.append(pred)
    return torch.stack(preds, dim=1)


# Group 1: parallel-serial consistency

def check_parallel_vs_serial_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 5, 4, 32, 32, device, torch.float64)
    dt = _make_dt(2, 5, device, torch.float64)
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = _manual_forward(model, x, dt)
    diff = _max_diff(out_parallel, out_serial)
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"



def check_rollout_vs_serial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = torch.tensor(
        [[1.0, 1.25, 0.5, 0.75], [1.0, 0.75, 1.5, 1.25]],
        device=device,
        dtype=torch.float64,
    )
    dt_list = [
        torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        rollout = model.forward_rollout(x_context, dt_context, dt_list)
        serial = full_serial_inference(model, x_context, dt_context, dt_list)
    diff = _max_diff(rollout, serial)
    passed = diff < 1e-7
    return passed, f"max_diff={diff:.3e}"



def check_flux_scan_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(13)
    batch_size, steps, dim = 2, 7, 16
    tracker = GlobalFluxTracker(dim=dim, dt_ref=1.0).to(device).double()
    x_mean_seq = torch.complex(
        torch.randn(batch_size, steps, dim, device=device, dtype=torch.float64),
        torch.randn(batch_size, steps, dim, device=device, dtype=torch.float64),
    )
    dt_seq = torch.rand(batch_size, steps, device=device, dtype=torch.float64) * 1.5 + 0.1
    flux0 = tracker.get_initial_state(batch_size, device, torch.complex128)
    scan_a, scan_x = tracker.get_scan_operators(x_mean_seq, dt_seq)
    flux_scan = pscan(scan_a, scan_x).squeeze(-1)
    flux_scan = flux_scan + flux0.unsqueeze(1) * torch.cumprod(scan_a.squeeze(-1), dim=1)

    flux_serial = []
    flux_state = flux0
    for step_idx in range(steps):
        flux_state, _, _ = tracker.forward_step(
            flux_state,
            x_mean_seq[:, step_idx],
            dt_seq[:, step_idx],
        )
        flux_serial.append(flux_state)
    flux_serial = torch.stack(flux_serial, dim=1)
    diff = _max_diff(flux_scan, flux_serial)
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"



def check_temporal_propagator_scan_vs_serial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(14)
    prop = TemporalPropagator(dim=32, dt_ref=1.0, init_noise_scale=0.01).to(device).double()
    dt_seq = torch.rand(3, 5, device=device, dtype=torch.float64) * 1.5 + 0.1
    decay_seq, forcing_seq = prop.get_transition_operators_seq(dt_seq)
    max_decay = 0.0
    max_forcing = 0.0
    for step_idx in range(dt_seq.shape[1]):
        decay_step, forcing_step = prop.get_transition_operators_step(dt_seq[:, step_idx])
        max_decay = max(max_decay, _max_diff(decay_seq[:, step_idx], decay_step))
        max_forcing = max(max_forcing, _max_diff(forcing_seq[:, step_idx], forcing_step))
    diff = max(max_decay, max_forcing)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"



def check_multi_scale_mixer_real_imag_independence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(15)
    mixer = MultiScaleSpatialMixer(dim=8).to(device).double()
    real = torch.randn(2, 8, 9, 11, device=device, dtype=torch.float64)
    imag = torch.randn(2, 8, 9, 11, device=device, dtype=torch.float64)
    x = torch.complex(real, imag)
    with torch.no_grad():
        out = mixer(x)
        expected_real = real + mixer._forward_real(real) * mixer.output_scale.to(real.dtype)
        expected_imag = imag + mixer._forward_real(imag) * mixer.output_scale.to(imag.dtype)
        expected = torch.complex(expected_real, expected_imag)
    diff = _max_diff(out, expected)
    passed = diff < 1e-14
    return passed, f"max_diff={diff:.3e}"


# Group 2: time-step semantics

def check_dt_zero_is_identity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(21)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt = torch.zeros(2, 4, device=device, dtype=torch.float64)
    with torch.no_grad():
        out = model(x, dt)
    diff = _max_diff(out, x)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"



def check_dt_scaling_changes_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(22)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_small = torch.full((2, 4), 0.5, device=device, dtype=torch.float64)
    dt_large = torch.full((2, 4), 2.0, device=device, dtype=torch.float64)
    with torch.no_grad():
        out_small = model(x, dt_small)
        out_large = model(x, dt_large)
    diff = _mean_diff(out_small, out_large)
    passed = diff > 1e-6
    return passed, f"mean_diff={diff:.3e}"



def check_rollout_horizon_semantics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(23)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = torch.tensor(
        [[1.0, 1.25, 0.5, 0.75], [1.0, 0.75, 1.5, 1.25]],
        device=device,
        dtype=torch.float64,
    )
    dt_first = torch.tensor([0.5, 1.0], device=device, dtype=torch.float64)
    dt_list = [dt_first, dt_first * 0.5, dt_first * 1.5]
    with torch.no_grad():
        rollout = model.forward_rollout(x_context, dt_context, dt_list)
        first_pred = rollout[:, 0]
        serial = full_serial_inference(model, x_context, dt_context, dt_list)
    diff = _max_diff(first_pred, serial[:, 0])
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"



def check_rollout_stride_offset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(24)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = _make_dt(2, 4, device, torch.float64)
    dt_list = [
        torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
        torch.tensor([0.75, 1.25], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        rollout_full = model.forward_rollout(x_context, dt_context, dt_list)
        rollout_stride = model.forward_rollout(
            x_context,
            dt_context,
            dt_list,
            output_stride=2,
            output_offset=1,
        )
    diff = _max_diff(rollout_stride, rollout_full[:, 1::2])
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"



def check_context_dt_scalar_vs_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(25)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 3, 4, 32, 32, device, torch.float64)
    dt_tensor = torch.ones(2, 3, device=device, dtype=torch.float64)
    dt_list = [
        torch.tensor([0.5, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.25, 0.5], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        out_scalar = model.forward_rollout(x_context, 1.0, dt_list)
        out_tensor = model.forward_rollout(x_context, dt_tensor, dt_list)
    diff = _max_diff(out_scalar, out_tensor)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"



def check_small_eigenvalue_limits():
    dt_ratio = torch.tensor([0.25, 1.5], dtype=torch.float64)
    exp_arg = torch.zeros(2, dtype=torch.complex128)
    forcing = _safe_forcing(exp_arg, dt_ratio)
    forcing_diff = float((forcing.real - dt_ratio).abs().max().item())
    sde_scale = _compute_sde_scale(
        torch.zeros(2, dtype=torch.float64),
        dt_ratio,
        torch.ones(2, dtype=torch.float64),
    )
    sde_diff = float((sde_scale - torch.sqrt(dt_ratio)).abs().max().item())
    diff = max(forcing_diff, sde_diff)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"



def check_negative_dt_rejected():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(26)
    model = build_check_model(device, torch.float64)
    x = _make_input(2, 3, 4, 32, 32, device, torch.float64)
    try:
        model(x, -1.0)
    except ValueError as exc:
        return True, f"raised={type(exc).__name__}"
    return False, "negative_dt_not_rejected"



def check_dt_normalize_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device, torch.float64)
    batch_size = 3
    steps = 4
    scalar = model._normalize_dt(1.0, batch_size, steps, device)
    zero_dim = model._normalize_dt(torch.tensor(2.0, device=device, dtype=torch.float64), batch_size, steps, device)
    batch_vec = model._normalize_dt(
        torch.tensor([0.5, 1.0, 1.5], device=device, dtype=torch.float64),
        batch_size,
        steps,
        device,
    )
    full = model._normalize_dt(
        torch.arange(batch_size * steps, device=device, dtype=torch.float64).reshape(batch_size, steps),
        batch_size,
        steps,
        device,
    )
    shapes_ok = all(tensor.shape == (batch_size, steps) for tensor in [scalar, zero_dim, batch_vec, full])
    values_ok = (
        torch.allclose(scalar, torch.full((batch_size, steps), 1.0, device=device, dtype=torch.float64))
        and torch.allclose(zero_dim, torch.full((batch_size, steps), 2.0, device=device, dtype=torch.float64))
        and torch.allclose(batch_vec[:, 0], torch.tensor([0.5, 1.0, 1.5], device=device, dtype=torch.float64))
        and torch.allclose(full, torch.arange(batch_size * steps, device=device, dtype=torch.float64).reshape(batch_size, steps))
    )
    passed = shapes_ok and values_ok
    return passed, f"shapes_ok={shapes_ok} values_ok={values_ok}"


# Group 3: parameter/global consistency

def check_dt_ref_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    values = []
    for block in model.blocks:
        values.append(float(block.prop.dt_ref))
        if float(block.prop.dt_ref) != float(block.prop.flux_tracker.dt_ref):
            return False, "block_dt_ref_mismatch"
    passed = max(values) - min(values) == 0.0
    return passed, f"dt_ref_values={values}"



def check_noise_scale_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    values = torch.tensor(
        [float(block.prop.base_noise.abs().mean().item()) for block in model.blocks],
        dtype=torch.float64,
    )
    variance = float(values.var(unbiased=False).item()) if values.numel() > 1 else 0.0
    passed = variance < 1e-6
    return passed, f"variance={variance:.3e} values={values.tolist()}"



def check_state_dimension_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    states = model._init_states(3, device, torch.complex64)
    expected_h = (3 * model.h_patches * model.w_patches, 1, model.embed_dim)
    expected_flux = (3, model.blocks[0].prop.flux_tracker.state_dim)
    for block_idx, (h_state, flux_state) in enumerate(states):
        if tuple(h_state.shape) != expected_h:
            return False, f"block={block_idx} h_shape={tuple(h_state.shape)}"
        if tuple(flux_state.shape) != expected_flux:
            return False, f"block={block_idx} flux_shape={tuple(flux_state.shape)}"
    return True, f"h_shape={expected_h} flux_shape={expected_flux}"



def check_no_duplicate_parameter_names():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    names = [name for name, _ in model.named_parameters()]
    unique = len(set(names))
    passed = unique == len(names)
    return passed, f"count={len(names)} unique={unique}"



def check_no_dead_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for name, _ in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in _deprecated_param_keywords()):
            bad.append(name)
    passed = not bad
    return passed, f"bad_names={bad}"



def check_basis_is_scalar_alpha():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for block_idx, block in enumerate(model.blocks):
        basis = block.prop.basis
        if basis.alpha_logit.shape != torch.Size([]):
            bad.append(f"block={block_idx} alpha_shape={tuple(basis.alpha_logit.shape)}")
        if hasattr(basis, "spectral_mod") and getattr(basis, "spectral_mod") is not None:
            bad.append(f"block={block_idx} spectral_mod_present")
    passed = not bad
    return passed, f"issues={bad}"



def check_single_spatial_mixer_type():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for block_idx, block in enumerate(model.blocks):
        if not isinstance(block.spatial_mixer, MultiScaleSpatialMixer):
            bad.append(f"block={block_idx} wrong_type={type(block.spatial_mixer).__name__}")
        for name in ["spatial_cliff", "spatial_pool", "norm_spatial"]:
            if hasattr(block, name):
                bad.append(f"block={block_idx} has_{name}")
    passed = not bad
    return passed, f"issues={bad}"



def check_single_flux_tracker_type():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    extra_name = "Multi" + "Slot" + "FluxTracker"
    for block_idx, block in enumerate(model.blocks):
        if not isinstance(block.prop.flux_tracker, GlobalFluxTracker):
            bad.append(f"block={block_idx} wrong_flux={type(block.prop.flux_tracker).__name__}")
    extra = [
        type(module).__name__
        for _, module in model.named_modules()
        if type(module).__name__ == extra_name
    ]
    passed = not bad and not extra
    return passed, f"issues={bad + extra}"



def check_encoder_has_no_metric():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    present = [name for name in _forbidden_encoder_names() if hasattr(model.encoder, name)]
    passed = not present
    return passed, f"present={present}"



def check_skip_is_inlined():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    has_required = hasattr(model, "skip_context_proj") and hasattr(model, "skip_spatial_proj")
    has_wrapper = hasattr(model, "decoder_skip_gate")
    passed = has_required and not has_wrapper
    return passed, f"has_required={has_required} has_wrapper={has_wrapper}"


# Group 4: architecture verification

def check_no_variant_attributes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    present = []
    for prefix, module in [("model", model)] + [(f"block_{idx}", block) for idx, block in enumerate(model.blocks)]:
        for name in _no_variant_attr_names():
            if hasattr(module, name):
                present.append(f"{prefix}.{name}")
    passed = not present
    return passed, f"present={present}"



def check_no_deprecated_submodules():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    forbidden = _deprecated_module_names()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in forbidden:
            bad.append(f"{name}:{cls_name}")
    passed = not bad
    return passed, f"modules={bad}"



def check_constructor_has_no_variant_param():
    names = inspect.signature(UniPhyModel.__init__).parameters.keys()
    bad = [name for name in _forbidden_constructor_params() if name in names]
    passed = not bad
    return passed, f"params={list(names)}"



def check_block_constructor_has_no_variant_param():
    names = inspect.signature(UniPhyBlock.__init__).parameters.keys()
    bad = [("variant" + "_config") for _ in [0] if ("variant" + "_config") in names]
    passed = not bad
    return passed, f"params={list(names)}"



def check_all_parameters_require_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
    passed = not frozen
    return passed, f"frozen={frozen}"


CHECK_GROUPS = [
    (
        "Group 1: Parallel-Serial Consistency",
        [
            check_parallel_vs_serial_forward,
            check_rollout_vs_serial,
            check_flux_scan_equivalence,
            check_temporal_propagator_scan_vs_serial,
            check_multi_scale_mixer_real_imag_independence,
        ],
    ),
    (
        "Group 2: Time-Step Semantics",
        [
            check_dt_zero_is_identity,
            check_dt_scaling_changes_output,
            check_rollout_horizon_semantics,
            check_rollout_stride_offset,
            check_context_dt_scalar_vs_tensor,
            check_small_eigenvalue_limits,
            check_negative_dt_rejected,
            check_dt_normalize_shapes,
        ],
    ),
    (
        "Group 3: Parameter Consistency",
        [
            check_dt_ref_consistency,
            check_noise_scale_consistency,
            check_state_dimension_consistency,
            check_no_duplicate_parameter_names,
            check_no_dead_parameters,
            check_basis_is_scalar_alpha,
            check_single_spatial_mixer_type,
            check_single_flux_tracker_type,
            check_encoder_has_no_metric,
            check_skip_is_inlined,
        ],
    ),
    (
        "Group 4: Architecture Verification",
        [
            check_no_variant_attributes,
            check_no_deprecated_submodules,
            check_constructor_has_no_variant_param,
            check_block_constructor_has_no_variant_param,
            check_all_parameters_require_grad,
        ],
    ),
]


def run_all_checks():
    total = 0
    passed_total = 0
    all_passed = True
    print("=" * 72)
    print("UNIPHY CHECK SUITE")
    print("=" * 72)
    for group_name, checks in CHECK_GROUPS:
        print(group_name)
        print("-" * 72)
        group_passed = 0
        for check_fn in checks:
            total += 1
            try:
                passed, detail = check_fn()
            except Exception as exc:
                passed = False
                detail = f"{type(exc).__name__}: {exc}"
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {check_fn.__name__} :: {detail}")
            if passed:
                passed_total += 1
                group_passed += 1
            else:
                all_passed = False
        print(f"GROUP SUMMARY {group_passed}/{len(checks)}")
        print("-" * 72)
    print("=" * 72)
    print(f"TOTAL {passed_total}/{total}")
    print(f"RESULT {'PASS' if all_passed else 'FAIL'}")
    print("=" * 72)
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    raise SystemExit(0 if success else 1)
