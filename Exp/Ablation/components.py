import torch
import torch.nn as nn

from Model.UniPhy.ModelUniPhy import UniPhyModel


class NoTimeDtWrapper(nn.Module):
    def __init__(self, base_model: UniPhyModel):
        super().__init__()
        self.base_model = base_model
        self.dt_ref = float(base_model.dt_ref)

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
        return [torch.full_like(dt, self.dt_ref) for dt in dt_list]

    def forward(self, x, dt, z=None, return_latent=False):
        return self.base_model(x, self._const_seq(dt), z=z, return_latent=return_latent)

    def forward_rollout(self, x_context, dt_context, dt_list, **kwargs):
        return self.base_model.forward_rollout(
            x_context,
            self._const_seq(dt_context),
            self._const_list(dt_list),
            **kwargs,
        )


class SingleScaleSpatialMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch_local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.branch_local(x.contiguous()) * self.output_scale


def apply_single_scale_mixer(model: UniPhyModel) -> UniPhyModel:
    device = next(model.parameters()).device
    dim = model.embed_dim
    for block in model.blocks:
        block.spatial_mixer = SingleScaleSpatialMixer(dim).to(device)
    return model


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
