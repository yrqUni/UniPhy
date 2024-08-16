import torch
import numpy as np

class LRU:
    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def compute_parallel(self, h, lamb, mask):
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        return h

    def lru_serial(self, h, lamb, mask):
        B, L, D = h.size()
        for l in range(1, L):
            h[:, l, :] += h[:, l - 1, :] * lamb * mask[:, l - 1].unsqueeze(-1)
        return h

B, L, D = 2, 8, 128
h = torch.randn(B, L, D)
lamb = torch.randn(1, D)
mask = torch.ones(B, L)

lru = LRU()
h_result_parallel = lru.compute_parallel(h.clone(), lamb.clone(), mask.clone())
h_result_serial = lru.lru_serial(h.clone(), lamb.clone(), mask.clone())
out = (h_result_parallel-h_result_serial).abs()
print(torch.max(out[~torch.isnan(out)]))
print(torch.allclose(h_result_parallel, h_result_serial))

##############################################

import torch
import numpy as np

class LRU:
    def lru_parallel(self, i, h, lamb, mask, B, L, C, S):
        l = 2 ** i
        h = h.reshape(B * L // l, l, C, S, S)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        mask_ = mask_[:, l // 2 - 1:l // 2]
        h2 = h2 + lamb * h1[:, -1:] * mask_.reshape(*mask_.shape, 1, 1, 1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def compute_parallel(self, h, lamb, mask):
        B, L, C, S, S = h.size()
        log2_L = int(np.ceil(np.log2(L)))
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, C, S)
        return h

    def lru_serial(self, h, lamb, mask):
        B, L, C, S, S = h.size()
        for l in range(1, L):
            mask_ = mask[:, l - 1]
            h[:, l] += h[:, l - 1] * lamb * mask_.reshape(*mask_.shape, 1, 1, 1)
        return h

B, L, C, S = 2, 8, 128, 1
h = torch.randn(B, L, C, S, S)
lamb = torch.randn(1, C, S, S)
mask = torch.ones(B, L)

lru = LRU()
h_result_parallel = lru.compute_parallel(h.clone(), lamb.clone(), mask.clone())
h_result_serial = lru.lru_serial(h.clone(), lamb.clone(), mask.clone())
out = (h_result_parallel-h_result_serial).abs()
print(torch.max(out[~torch.isnan(out)]))
print(torch.allclose(h_result_parallel, h_result_serial))
