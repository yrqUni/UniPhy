import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.UniPhy.ModelUniPhy import UniPhyModel, _dt_is_zero, _expand_batch_mask


class NoTimeDtWrapper(nn.Module):

    def __init__(self, base_model: UniPhyModel):
        super().__init__()
        self.base_model = base_model
        self.dt_ref = float(base_model.blocks[0].prop.dt_ref)

    @property
    def blocks(self):
        return self.base_model.blocks

    @property
    def embed_dim(self):
        return self.base_model.embed_dim

    @property
    def depth(self):
        return self.base_model.depth

    @property
    def h_patches(self):
        return self.base_model.h_patches

    @property
    def w_patches(self):
        return self.base_model.w_patches

    def _const_seq(self, dt):
        return torch.full_like(dt, self.dt_ref)

    def _const_list(self, dt_list):
        return [torch.full_like(d, self.dt_ref) for d in dt_list]

    def forward(self, x, dt, z=None, return_latent=False):
        return self.base_model(x, self._const_seq(dt), z=z, return_latent=return_latent)

    def forward_rollout(self, x_context, dt_context, dt_list, **kwargs):
        return self.base_model.forward_rollout(
            x_context,
            self._const_seq(dt_context),
            self._const_list(dt_list),
            **kwargs,
        )

    def sample_noise(self, x):
        return self.base_model.sample_noise(x)

    def sample_block_noises(self, x):
        return self.base_model.sample_block_noises(x)

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        return self.base_model.sample_rollout_noise(batch_size, steps, device, dtype)


def apply_elman_rnn(model, dim):
    device = next(model.parameters()).device
    new_blocks = []
    for block in model.blocks:
        new_blocks.append(_RealElmanBlock(block, dim).to(device))
    new_blocks = nn.ModuleList(new_blocks)
    model.blocks = new_blocks
    return model


class _RealElmanBlock(nn.Module):

    def __init__(self, base_block, dim):
        super().__init__()
        self.base = base_block
        self.prop = base_block.prop
        self.spatial_mixer = base_block.spatial_mixer
        self.rnn_h = nn.Linear(dim, dim, bias=False)
        self.rnn_u = nn.Linear(dim, dim, bias=True)
        nn.init.orthogonal_(self.rnn_h.weight)
        nn.init.xavier_uniform_(self.rnn_u.weight)

    def _elman_h(self, h_prev_hw, forcing):
        return torch.tanh(self.rnn_h(h_prev_hw) + self.rnn_u(forcing))

    def forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
        del flux_prev, noise_seq
        batch_size, steps, dim, height, width = x.shape
        x_real = x.real if torch.is_complex(x) else x
        x_flat = x_real.contiguous().reshape(batch_size * steps, dim, height, width)
        x_flat = self.spatial_mixer(x_flat)
        x_seq = x_flat.reshape(batch_size, steps, dim, height, width)
        x_hw = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        rows = []
        for t in range(steps):
            h_cur = self._elman_h(h_cur, x_hw[:, t])
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq[:, t]), h_cur.ndim)
            h_cur = torch.where(zero_mask, x_hw[:, t], h_cur)
            rows.append(h_cur)
        seq = torch.stack(rows, dim=1)
        lead_time = torch.cumsum(dt_seq, dim=1)
        combined = self.base._decode_sequence(seq, lead_time)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x_real, combined)
        h_out = seq[:, -1].reshape(batch_size * height * width, 1, dim)
        flux_out = seq[:, -1].mean(dim=(1, 2))
        return combined, h_out, flux_out

    def forward_step(
        self,
        x_curr,
        x_next,
        h_prev,
        dt_step,
        dt_next,
        flux_prev,
        noise_step=None,
        lead_time=None,
    ):
        del x_next, dt_next, flux_prev, noise_step
        batch_size, dim, height, width = x_curr.shape
        x_real = x_curr.real if torch.is_complex(x_curr) else x_curr
        x_mix = self.spatial_mixer(x_real.contiguous())
        x_hw = x_mix.permute(0, 2, 3, 1).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        h_next = self._elman_h(h_cur, x_hw)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), h_next.ndim)
        h_next = torch.where(zero_mask, x_hw, h_next)
        if lead_time is None:
            lead_time = dt_step
        z_next = self.base._decode_sequence(h_next.unsqueeze(1), lead_time.unsqueeze(1))[
            :, 0
        ]
        z_next = torch.where(
            _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim),
            x_real,
            z_next,
        )
        flux_next = h_next.mean(dim=(1, 2))
        return z_next, h_next.reshape(batch_size * height * width, 1, dim), flux_next


class DeterministicWrapper(nn.Module):

    def __init__(self, base_model: UniPhyModel):
        super().__init__()
        self.base_model = base_model

    @property
    def blocks(self):
        return self.base_model.blocks

    @property
    def embed_dim(self):
        return self.base_model.embed_dim

    @property
    def depth(self):
        return self.base_model.depth

    @property
    def h_patches(self):
        return self.base_model.h_patches

    @property
    def w_patches(self):
        return self.base_model.w_patches

    def forward(self, x, dt, z=None, return_latent=False):
        return self.base_model(x, dt, z=None, return_latent=return_latent)

    def forward_rollout(self, x_context, dt_context, dt_list, **kwargs):
        kwargs.pop("z_context", None)
        kwargs.pop("z_rollout", None)
        return self.base_model.forward_rollout(
            x_context, dt_context, dt_list, z_context=None, z_rollout=None, **kwargs
        )

    def sample_noise(self, x):
        return None

    def sample_block_noises(self, x):
        return [None] * self.base_model.depth

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        return None


class SingleScaleSpatialMixer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.branch_local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_complex):
        if torch.is_complex(x_complex):
            real = self.branch_local(x_complex.real)
            imag = self.branch_local(x_complex.imag)
            return x_complex + torch.complex(real, imag) * self.output_scale
        return x_complex + self.branch_local(x_complex) * self.output_scale


def apply_single_scale_mixer(model: UniPhyModel) -> UniPhyModel:
    device = next(model.parameters()).device
    dim = model.embed_dim
    for block in model.blocks:
        block.spatial_mixer = SingleScaleSpatialMixer(dim).to(device)
    return model


def apply_fixed_decay(model: UniPhyModel) -> UniPhyModel:
    for block in model.blocks:
        old = block.prop
        with torch.no_grad():
            target = torch.tensor(1.0, device=old.decay_logit.device)
            old.decay_logit.fill_(torch.log(torch.expm1(target)).item())
        old.decay_logit.requires_grad_(False)
    return model


def apply_etd1_integrator(model: UniPhyModel) -> UniPhyModel:
    for block in model.blocks:
        block.__class__ = _RealEulerBlock
    return model


class _RealEulerBlock(nn.Module):

    def _decode_sequence(self, seq, lead_time):
        del lead_time
        return seq.permute(0, 1, 4, 2, 3).contiguous()

    def _euler_factor(self, dt, target_ndim):
        dt_ratio = dt / self.prop.dt_ref
        while dt_ratio.ndim < target_ndim:
            dt_ratio = dt_ratio.unsqueeze(-1)
        lam = torch.nn.functional.softplus(self.prop.decay_logit).to(dt_ratio.dtype)
        while lam.ndim < target_ndim:
            lam = lam.unsqueeze(0)
        return lam * dt_ratio

    def forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
        del flux_prev
        batch_size, steps, dim, height, width = x.shape
        x_real = x.real if torch.is_complex(x) else x
        x_flat = x_real.contiguous().reshape(batch_size * steps, dim, height, width)
        x_flat = self.spatial_mixer(x_flat)
        x_seq = x_flat.reshape(batch_size, steps, dim, height, width)
        x_hw = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        rows = []
        for t in range(steps):
            dt_t = dt_seq[:, t]
            factor = self._euler_factor(dt_t, 4)
            noise_t = None if noise_seq is None else noise_seq[:, t]
            h_next = h_cur + factor * (x_hw[:, t] - h_cur)
            h_next = h_next + self.prop.noise(noise_t, h_next.shape, dt_t)
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_t), h_next.ndim)
            h_cur = torch.where(zero_mask, x_hw[:, t], h_next)
            rows.append(h_cur)
        seq = torch.stack(rows, dim=1)
        lead_time = torch.cumsum(dt_seq, dim=1)
        combined = self._decode_sequence(seq, lead_time)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x_real, combined)
        h_out = seq[:, -1].reshape(batch_size * height * width, 1, dim)
        flux_out = seq[:, -1].mean(dim=(1, 2))
        return combined, h_out, flux_out

    def forward_step(
        self,
        x_curr,
        x_next,
        h_prev,
        dt_step,
        dt_next,
        flux_prev,
        noise_step=None,
        lead_time=None,
    ):
        del x_next, dt_next, flux_prev
        batch_size, dim, height, width = x_curr.shape
        x_real = x_curr.real if torch.is_complex(x_curr) else x_curr
        x_mix = self.spatial_mixer(x_real.contiguous())
        x_hw = x_mix.permute(0, 2, 3, 1).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        factor = self._euler_factor(dt_step, 4)
        h_next = h_cur + factor * (x_hw - h_cur)
        h_next = h_next + self.prop.noise(noise_step, h_next.shape, dt_step)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), h_next.ndim)
        h_next = torch.where(zero_mask, x_hw, h_next)
        if lead_time is None:
            lead_time = dt_step
        z_next = self._decode_sequence(h_next.unsqueeze(1), lead_time.unsqueeze(1))[
            :, 0
        ]
        z_next = torch.where(
            _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim),
            x_real,
            z_next,
        )
        flux_next = h_next.mean(dim=(1, 2))
        return z_next, h_next.reshape(batch_size * height * width, 1, dim), flux_next


def apply_fixed_scale_weights(model: UniPhyModel) -> UniPhyModel:
    for block in model.blocks:
        mixer = getattr(block, "spatial_mixer", None)
        gate = getattr(mixer, "mix_gate", None)
        if gate is None:
            continue
        with torch.no_grad():
            for module in gate.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        for param in gate.parameters():
            param.requires_grad_(False)
    return model


def apply_readout_residual(model: UniPhyModel) -> UniPhyModel:
    dim = model.embed_dim
    hidden = int(dim * 4)
    device = next(model.parameters()).device
    for block in model.blocks:
        block.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        ).to(device)
        block.output_scale = nn.Parameter(torch.tensor(0.1, device=device))
        block.readout_decay_logit = nn.Parameter(torch.tensor(0.0, device=device))
        block.__class__ = _RealReadoutResidualBlock
    return model


class _RealReadoutResidualBlock(_RealEulerBlock):

    def _readout_weight(self, lead_time, target_ndim):
        rate = F.softplus(self.readout_decay_logit).to(lead_time.dtype)
        weight = torch.exp(-rate * (lead_time / self.prop.dt_ref))
        while weight.ndim < target_ndim:
            weight = weight.unsqueeze(-1)
        return weight

    def _decode_sequence(self, seq, lead_time):
        batch_size, steps, height, width, dim = seq.shape
        flat = seq.permute(0, 1, 4, 2, 3).contiguous().reshape(
            batch_size * steps,
            dim,
            height,
            width,
        )
        weight = self._readout_weight(lead_time, 2).reshape(batch_size * steps, 1)
        while weight.ndim < flat.ndim:
            weight = weight.unsqueeze(-1)
        flat = flat + self.ffn(flat) * self.output_scale * weight
        return flat.reshape(batch_size, steps, dim, height, width)


def apply_constant_readout(model: UniPhyModel) -> UniPhyModel:
    apply_readout_residual(model)
    for block in model.blocks:
        decay = getattr(block, "readout_decay_logit", None)
        if decay is None:
            continue
        with torch.no_grad():
            decay.fill_(-20.0)
        decay.requires_grad_(False)
    return model
