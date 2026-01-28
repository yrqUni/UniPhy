import torch
import torch.nn as nn
import torch.nn.functional as F


class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand, num_experts, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * expand
        self.num_experts = num_experts

        self.fc1_re = nn.Linear(dim, self.hidden_dim)
        self.fc1_im = nn.Linear(dim, self.hidden_dim)

        self.dw_conv_re = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
            bias=False
        )
        self.dw_conv_im = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
            bias=False
        )

        self.fc2_re = nn.Linear(self.hidden_dim, dim)
        self.fc2_im = nn.Linear(self.hidden_dim, dim)

        self.route_gate = nn.Linear(dim * 2, num_experts)
        self.dropout = nn.Dropout(dropout)

        self.out_conv_re = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.out_conv_im = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, D, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2).reshape(B * T, H, W, D)
        else:
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1)

        x_re = x.real
        x_im = x.imag

        x_cat = torch.cat([x_re, x_im], dim=-1)
        router_logits = self.route_gate(x_cat)
        route_weights = F.softmax(router_logits, dim=-1)

        prob_mean = route_weights.mean(dim=(0, 1, 2))
        self.aux_loss = (self.num_experts * (prob_mean ** 2).sum()) * 0.01

        h_re = self.fc1_re(x_re) - self.fc1_im(x_im)
        h_im = self.fc1_re(x_im) + self.fc1_im(x_re)

        h_re = h_re.permute(0, 3, 1, 2)
        h_im = h_im.permute(0, 3, 1, 2)

        h_re = self.dw_conv_re(h_re)
        h_im = self.dw_conv_im(h_im)

        h_re = h_re.permute(0, 2, 3, 1)
        h_im = h_im.permute(0, 2, 3, 1)

        h_re = F.gelu(h_re)
        h_im = F.gelu(h_im)

        h_re = self.dropout(h_re)
        h_im = self.dropout(h_im)

        out_re = self.fc2_re(h_re) - self.fc2_im(h_im)
        out_im = self.fc2_re(h_im) + self.fc2_im(h_re)

        out_complex = torch.complex(out_re, out_im)

        route_weights = route_weights.unsqueeze(-1)
        out_weighted = out_complex * route_weights.sum(dim=-2, keepdim=True)

        out_permuted = out_weighted.permute(0, 3, 1, 2)
        final_re = self.out_conv_re(out_permuted.real)
        final_im = self.out_conv_im(out_permuted.imag)
        out_final = torch.complex(final_re, final_im)

        if is_5d:
            out_final = out_final.reshape(B, T, D, H, W)

        return out_final

