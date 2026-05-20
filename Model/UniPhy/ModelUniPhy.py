import torch
import torch.nn as nn

from .UniPhyIO import UniPhyRealDecoder, UniPhyRealEncoder
from .UniPhyOps import RealMultiScaleSpatialMixer


def _dt_is_zero(dt, tol=1e-12):
    return dt.abs() <= tol


def _dt_has_negative(dt, tol=1e-12):
    return bool((dt < -tol).any().item())


def _dt_has_nonfinite(dt):
    return not bool(torch.isfinite(dt).all().item())


def _expand_batch_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


class UniPhyBlock(nn.Module):
    def __init__(self, dim, expand=None, dt_ref=6.0):
        super().__init__()
        del expand
        self.dt_ref = float(dt_ref)
        self.spatial_mixer = RealMultiScaleSpatialMixer(dim)
        self.rnn_h = nn.Linear(dim, dim, bias=False)
        self.rnn_u = nn.Linear(dim, dim, bias=True)
        self.h0 = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.rnn_h.weight)
        nn.init.xavier_uniform_(self.rnn_u.weight)

    def get_initial_h(self, batch_size, height, width, device, dtype):
        h0 = self.h0.to(device=device, dtype=dtype)
        return h0.reshape(1, 1, 1, -1).expand(batch_size, height, width, -1)

    def _decode_sequence(self, seq, lead_time=None):
        del lead_time
        return seq.permute(0, 1, 4, 2, 3).contiguous()

    def _mix_input(self, x):
        x_real = x.real if torch.is_complex(x) else x
        return self.spatial_mixer(x_real.contiguous())

    def _step_h(self, h_prev, forcing):
        return torch.tanh(self.rnn_h(h_prev) + self.rnn_u(forcing))

    def forward(self, x, h_prev, dt_seq):
        batch_size, steps, dim, height, width = x.shape
        x_real = x.real if torch.is_complex(x) else x
        x_flat = x_real.contiguous().reshape(batch_size * steps, dim, height, width)
        x_flat = self.spatial_mixer(x_flat)
        x_seq = x_flat.reshape(batch_size, steps, dim, height, width)
        x_hw = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        rows = []
        for step_idx in range(steps):
            dt_step = dt_seq[:, step_idx]
            h_next = self._step_h(h_cur, x_hw[:, step_idx])
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), h_next.ndim)
            h_cur = torch.where(zero_mask, x_hw[:, step_idx], h_next)
            rows.append(h_cur)
        seq = torch.stack(rows, dim=1)
        combined = self._decode_sequence(seq, torch.cumsum(dt_seq, dim=1))
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x_real, combined)
        h_out = seq[:, -1].reshape(batch_size * height * width, 1, dim)
        return combined, h_out

    def forward_step(self, x_curr, h_prev, dt_step, lead_time=None):
        del lead_time
        batch_size, dim, height, width = x_curr.shape
        x_real = x_curr.real if torch.is_complex(x_curr) else x_curr
        x_mix = self._mix_input(x_real)
        x_hw = x_mix.permute(0, 2, 3, 1).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        h_next = self._step_h(h_cur, x_hw)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), h_next.ndim)
        h_next = torch.where(zero_mask, x_hw, h_next)
        z_next = self._decode_sequence(h_next.unsqueeze(1), None)[:, 0]
        z_next = torch.where(
            _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim),
            x_real,
            z_next,
        )
        return z_next, h_next.reshape(batch_size * height * width, 1, dim)


class UniPhyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        expand,
        depth,
        patch_size,
        img_height,
        img_width,
        dt_ref=6.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.dt_ref = float(dt_ref)
        ph, pw = tuple(patch_size)
        self.h_patches = (int(img_height) + (ph - int(img_height) % ph) % ph) // ph
        self.w_patches = (int(img_width) + (pw - int(img_width) % pw) % pw) // pw
        self.encoder = UniPhyRealEncoder(
            in_ch=int(in_channels),
            embed_dim=int(embed_dim),
            patch_size=tuple(patch_size),
            img_height=int(img_height),
            img_width=int(img_width),
        )
        self.decoder = UniPhyRealDecoder(
            out_ch=int(out_channels),
            latent_dim=int(embed_dim),
            patch_size=tuple(patch_size),
            model_channels=int(embed_dim),
            img_height=int(img_height),
            img_width=int(img_width),
        )
        self.skip_context_proj = nn.Sequential(
            nn.Linear(int(embed_dim) * 2, int(embed_dim)),
            nn.SiLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )
        self.skip_spatial_proj = nn.Sequential(
            nn.Conv2d(int(embed_dim), int(embed_dim), 3, padding=1, groups=int(embed_dim)),
            nn.SiLU(),
            nn.Conv2d(int(embed_dim), int(embed_dim), 1),
        )
        nn.init.zeros_(self.skip_spatial_proj[-1].weight)
        nn.init.zeros_(self.skip_spatial_proj[-1].bias)
        self.blocks = nn.ModuleList(
            [
                UniPhyBlock(
                    dim=int(embed_dim),
                    expand=expand,
                    dt_ref=float(dt_ref),
                )
                for _ in range(int(depth))
            ]
        )
        self._init_encoder_weights()

    def _init_encoder_weights(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                nonlinearity = "linear" if "stem" in name else "relu"
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity=nonlinearity)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_states(self, batch_size, device, dtype):
        height, width, dim = self.h_patches, self.w_patches, self.embed_dim
        states = []
        for block in self.blocks:
            h0 = block.get_initial_h(batch_size, height, width, device, dtype)
            states.append(h0.reshape(batch_size * height * width, 1, dim))
        return states

    def _normalize_dt(self, dt, batch_size, steps, device):
        dtype = next(self.parameters()).dtype
        dt = dt.detach().to(device=device, dtype=dtype)
        if dt.ndim != 2:
            raise ValueError(f"dt_must_be_2d: got_ndim={dt.ndim}")
        rows, cols = dt.shape
        if (rows, cols) != (batch_size, steps):
            raise ValueError(f"dt_shape_mismatch: expected={(batch_size, steps)} got={(rows, cols)}")
        return dt

    def _normalize_step_dt(self, dt, batch_size, device):
        dtype = next(self.parameters()).dtype
        dt = dt.detach().to(device=device, dtype=dtype)
        if dt.ndim != 1:
            raise ValueError(f"dt_step_must_be_1d: got_ndim={dt.ndim}")
        if dt.shape[0] != batch_size:
            raise ValueError(f"dt_step_shape_mismatch: expected={batch_size} got={dt.shape[0]}")
        return dt

    def _validate_dt(self, dt):
        if _dt_has_nonfinite(dt):
            raise ValueError("dt_must_be_finite")
        if _dt_has_negative(dt):
            raise ValueError("dt_must_be_nonnegative")

    def _split_rollout_steps(self, dt_steps, allow_split=True):
        if not allow_split:
            return dt_steps, list(range(len(dt_steps)))
        split_steps = []
        output_indices = []
        for dt_step in dt_steps:
            ratio = dt_step.detach() / self.dt_ref
            rounded = torch.round(ratio)
            can_split = bool(
                torch.isfinite(ratio).all().item()
                and (ratio - rounded).abs().max().item() < 1e-6
                and rounded.min().item() == rounded.max().item()
                and rounded[0].item() > 1
            )
            if can_split:
                parts = int(rounded[0].item())
                sub_step = dt_step / parts
                for _ in range(parts):
                    split_steps.append(sub_step)
            else:
                split_steps.append(dt_step)
            output_indices.append(len(split_steps) - 1)
        return split_steps, output_indices

    def _skip_gate_4d(self, z_dec, z_skip):
        dec_ctx = torch.cat(
            [z_dec.mean(dim=(-2, -1)), z_skip.mean(dim=(-2, -1))],
            dim=1,
        )
        gate = self.skip_context_proj(dec_ctx).to(z_dec.dtype)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        delta_mag = (z_skip - z_dec).abs()
        gate = torch.sigmoid(gate + self.skip_spatial_proj(delta_mag).to(z_dec.dtype))
        return z_dec * (1.0 - gate) + z_skip * gate

    def _apply_decoder_skip(self, z_dec, z_skip):
        if z_skip is None:
            return z_dec
        batch_size, steps, dim, height, width = z_dec.shape
        z_dec_flat = z_dec.contiguous().view(batch_size * steps, dim, height, width)
        z_skip_flat = z_skip.contiguous().view(batch_size * steps, dim, height, width)
        z_out_flat = self._skip_gate_4d(z_dec_flat, z_skip_flat)
        return z_out_flat.view(batch_size, steps, dim, height, width)

    def forward(self, x, dt, z=None, return_latent=False):
        del z
        batch_size, steps = x.shape[0], x.shape[1]
        dt_seq = self._normalize_dt(dt, batch_size, steps, x.device)
        self._validate_dt(dt_seq)
        latent = self.encoder(x)
        z_skip = latent
        states = self._init_states(batch_size, x.device, latent.dtype)
        for block_idx, block in enumerate(self.blocks):
            latent, h_next = block(latent, states[block_idx], dt_seq)
            states[block_idx] = h_next
        latent = self._apply_decoder_skip(latent, z_skip)
        out = self.decoder(latent)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), out.ndim)
        out = torch.where(zero_mask, x, out)
        if return_latent:
            return out, latent
        return out

    def _advance_latent(self, z_curr, states, dt_step, lead_time=None):
        z_running = z_curr
        new_states = []
        for block_idx, block in enumerate(self.blocks):
            z_running, h_next = block.forward_step(
                z_running,
                states[block_idx],
                dt_step,
                lead_time=lead_time,
            )
            new_states.append(h_next)
        return z_running, new_states

    def forward_rollout(
        self,
        x_context,
        dt_context,
        dt_list,
        z_context=None,
        z_rollout=None,
        chunk_size=1,
        output_stride=1,
        output_offset=0,
    ):
        del z_context, z_rollout, chunk_size
        batch_size, steps_in = x_context.shape[0], x_context.shape[1]
        device = x_context.device
        dt_ctx_seq = self._normalize_dt(dt_context, batch_size, steps_in, device)
        self._validate_dt(dt_ctx_seq)
        z_ctx = self.encoder(x_context)
        states = self._init_states(batch_size, device, z_ctx.dtype)
        requested_dt = [self._normalize_step_dt(dt_step, batch_size, device) for dt_step in dt_list]
        for dt_step in requested_dt:
            self._validate_dt(dt_step)
        dt_steps, output_indices = self._split_rollout_steps(requested_dt, allow_split=True)
        if steps_in > 1:
            for step_idx in range(steps_in - 1):
                dt_step = dt_ctx_seq[:, step_idx + 1]
                z_running = z_ctx[:, step_idx]
                new_states = []
                for block_idx, block in enumerate(self.blocks):
                    z_running, h_next = block.forward_step(
                        z_running,
                        states[block_idx],
                        dt_step,
                        lead_time=dt_step,
                    )
                    new_states.append(h_next)
                states = new_states
        z_curr = z_ctx[:, -1]
        x_curr = x_context[:, -1]
        output_stride = max(1, int(output_stride))
        output_offset = int(output_offset)
        output_index_map = {
            internal_idx: output_idx for output_idx, internal_idx in enumerate(output_indices)
        }
        preds = []
        lead_accum = torch.zeros_like(dt_steps[0]) if dt_steps else None
        for step_idx, dt_step in enumerate(dt_steps):
            lead_accum = lead_accum + dt_step
            step_skip = z_curr
            z_running, states = self._advance_latent(
                z_curr,
                states,
                dt_step,
                lead_time=lead_accum,
            )
            z_curr = self._apply_decoder_skip(z_running.unsqueeze(1), step_skip.unsqueeze(1))[:, 0]
            x_pred = self.decoder(z_curr.unsqueeze(1))[:, 0]
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), x_pred.ndim)
            x_pred = torch.where(zero_mask, x_curr, x_pred)
            x_curr = x_pred
            output_idx = output_index_map.get(step_idx)
            if output_idx is not None and output_idx >= output_offset and (output_idx - output_offset) % output_stride == 0:
                preds.append(x_pred)
        if not preds:
            shape = (batch_size, 0) + tuple(x_context.shape[2:])
            return x_context.new_empty(shape)
        return torch.stack(preds, dim=1)
