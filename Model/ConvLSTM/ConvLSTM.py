import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        depth,
        patch_size,
        img_height,
        img_width,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.patch_size = tuple(patch_size)
        self.encoder = nn.Conv2d(
            in_channels,
            embed_dim,
            self.patch_size,
            stride=self.patch_size,
        )
        self.cells = nn.ModuleList(
            [ConvLSTMCell(embed_dim, embed_dim) for _ in range(depth)]
        )
        self.decoder = nn.ConvTranspose2d(
            embed_dim,
            out_channels,
            self.patch_size,
            stride=self.patch_size,
        )

    def _encode(self, x):
        ph, pw = self.patch_size
        h, w = x.shape[-2:]
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return self.encoder(x)

    def _decode(self, h):
        y = self.decoder(h)
        return y[..., : self.img_height, : self.img_width]

    def _zero_state(self, batch_size, height, width, device, dtype):
        return [
            (
                torch.zeros(
                    batch_size,
                    self.embed_dim,
                    height,
                    width,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(
                    batch_size,
                    self.embed_dim,
                    height,
                    width,
                    device=device,
                    dtype=dtype,
                ),
            )
            for _ in self.cells
        ]

    def sample_noise(self, x):
        return None

    def sample_block_noises(self, x):
        return [None] * self.depth

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        del batch_size, steps, device, dtype
        return None

    def _step_latent(self, z, states):
        new_states = []
        h = z
        for cell, state in zip(self.cells, states):
            h, c = cell(h, state)
            new_states.append((h, c))
        return h, new_states

    def forward(self, x, dt, z=None, return_latent=False):
        del dt, z
        b, t, c, h, w = x.shape
        z0 = self._encode(x[:, 0])
        states = self._zero_state(b, z0.shape[-2], z0.shape[-1], z0.device, z0.dtype)
        outs = []
        for i in range(t):
            zi = self._encode(x[:, i])
            h_state, states = self._step_latent(zi, states)
            outs.append(self._decode(h_state))
        out = torch.stack(outs, dim=1)
        if return_latent:
            return out, None
        return out

    def forward_rollout(self, x_context, dt_context, dt_list, **kwargs):
        del dt_context, kwargs
        b = x_context.shape[0]
        z0 = self._encode(x_context[:, 0])
        states = self._zero_state(b, z0.shape[-2], z0.shape[-1], z0.device, z0.dtype)
        for i in range(x_context.shape[1]):
            zi = self._encode(x_context[:, i])
            h_state, states = self._step_latent(zi, states)
        x_curr = self._decode(h_state)
        preds = []
        for _ in dt_list:
            zi = self._encode(x_curr)
            h_state, states = self._step_latent(zi, states)
            x_curr = self._decode(h_state)
            preds.append(x_curr)
        return torch.stack(preds, dim=1)
