import torch


def _validate_pscan_inputs(a, x):
    squeeze_output = x.ndim == 4
    if squeeze_output:
        x = x.unsqueeze(-1)
    if x.ndim != 5:
        raise ValueError(f"unsupported_x_rank={x.ndim}")
    if a.ndim not in {4, 5}:
        raise ValueError(f"unsupported_a_rank={a.ndim}")
    if a.shape[:3] != x.shape[:3]:
        raise ValueError(f"shape_mismatch_a={a.shape} x={x.shape}")
    is_diag = a.ndim == 4
    if is_diag:
        if a.shape[-1] != x.shape[-2]:
            raise ValueError(f"diag_state_mismatch_a={a.shape} x={x.shape}")
    else:
        if a.shape[-1] != a.shape[-2]:
            raise ValueError(f"nonsquare_transition={a.shape}")
        if a.shape[-1] != x.shape[-2]:
            raise ValueError(f"matrix_state_mismatch_a={a.shape} x={x.shape}")
    return a, x, is_diag, squeeze_output


def _scan_rows(x, advance_fn):
    state = torch.zeros_like(x[:, 0])
    rows = []
    for step in range(x.shape[1]):
        state = advance_fn(step, state)
        rows.append(state)
    return torch.stack(rows, dim=1)


def pscan(a, x):
    a, x, is_diag, squeeze_output = _validate_pscan_inputs(a, x)
    if is_diag:
        y = _scan_rows(
            x,
            lambda step, state: a[:, step].unsqueeze(-1) * state + x[:, step],
        )
    else:
        y = _scan_rows(
            x,
            lambda step, state: torch.einsum(
                "bcij,bcjk->bcik",
                a[:, step],
                state,
            )
            + x[:, step],
        )
    return y.squeeze(-1) if squeeze_output else y
