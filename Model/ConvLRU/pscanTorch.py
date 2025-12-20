import math
import torch
import torch.nn.functional as F

def npo2(length):
    return 1 if length <= 1 else 2 ** math.ceil(math.log2(length))

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        B = A.size(0)
        L = A.size(1)
        
        A = A.view(B, L, -1)
        X = X.view(B, L, -1)
        D = A.size(2)
        
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, D)
            Xa = Xa.view(B, T // 2, 2, D)
            
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            
            Aa = Aa[:, :, 1]
            Xa = Xa[:, :, 1]
            
        if Xa.size(1) == 4:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            Aa[:, 1].mul_(Aa[:, 0])
            Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
        elif Xa.size(1) == 2:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            return
        else:
            return
            
        Aa = A[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        
        Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 1]))
        Aa[:, 2].mul_(Aa[:, 1])
        
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 2 ** k - 1:L:2 ** k]
            Xa = X[:, 2 ** k - 1:L:2 ** k]
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, D)
            Xa = Xa.view(B, T // 2, 2, D)
            
            Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B = A.size(0)
        L = A.size(1)
        
        A = A.view(B, L, -1)
        X = X.view(B, L, -1)
        D = A.size(2)
        
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, D)
            Xa = Xa.view(B, T // 2, 2, D)
            
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Aa[:, :, 0].mul_(Aa[:, :, 1])
            
            Aa = Aa[:, :, 0]
            Xa = Xa[:, :, 0]
            
        if Xa.size(1) == 4:
            Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
            Aa[:, 2].mul_(Aa[:, 3])
            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
        elif Xa.size(1) == 2:
            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
            return
        else:
            return
            
        Aa = A[:, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, 0:L:2 ** (num_steps - 2)]
        
        Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 2]))
        Aa[:, 1].mul_(Aa[:, 2])
        
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 0:L:2 ** k]
            Xa = X[:, 0:L:2 ** k]
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, D)
            Xa = Xa.view(B, T // 2, 2, D)
            
            Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
            Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        
        if A_in.is_complex():
            calc_dtype = torch.complex64
        else:
            calc_dtype = torch.float32

        if L == npo2(L):
            A = A_in.clone().to(dtype=calc_dtype).contiguous()
            X = X_in.clone().to(dtype=calc_dtype).contiguous()
        else:
            def pad_dim1(t):
                pads = [0] * (t.ndim * 2)
                pads[2 * (t.ndim - 2) + 1] = npo2(L) - L
                return F.pad(t, tuple(pads), "constant", 0)
                
            A = pad_dim1(A_in).to(dtype=calc_dtype).contiguous()
            X = pad_dim1(X_in).to(dtype=calc_dtype).contiguous()
            
        PScan.pscan(A, X)
        
        ctx.save_for_backward(A_in, X)
        ctx.calc_dtype = calc_dtype
        return X[:, :L].to(dtype=A_in.dtype)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, H_pad = ctx.saved_tensors
        calc_dtype = ctx.calc_dtype
        L = grad_output_in.size(1)
        
        if L == npo2(L):
            grad_output = grad_output_in.clone().to(dtype=calc_dtype).contiguous()
            A_pad = A_in.clone().to(dtype=calc_dtype).contiguous()
        else:
            def pad_dim1(t):
                pads = [0] * (t.ndim * 2)
                pads[2 * (t.ndim - 2) + 1] = npo2(L) - L
                return F.pad(t, tuple(pads), "constant", 0)
                
            grad_output = pad_dim1(grad_output_in).to(dtype=calc_dtype).contiguous()
            A_pad = pad_dim1(A_in).to(dtype=calc_dtype).contiguous()
            
        slices_main = [slice(None)] * A_pad.ndim
        slices_main[1] = slice(None, -1)
        slices_first = [slice(None)] * A_pad.ndim
        slices_first[1] = slice(0, 1)
        
        A_shift = torch.cat([
            torch.zeros_like(A_pad[tuple(slices_first)]), 
            A_pad[tuple(slices_main)]
        ], dim=1)
        
        PScan.pscan_rev(A_shift.conj(), grad_output)
        gradX = grad_output[:, :L]
        
        H_prev = torch.cat([
            torch.zeros_like(H_pad[tuple(slices_first)]), 
            H_pad[tuple(slices_main)]
        ], dim=1)
        
        gradA_full = H_prev.conj() * grad_output
        gradA = gradA_full[:, :L]
        
        return gradA.to(dtype=A_in.dtype), gradX.to(dtype=A_in.dtype)

pscan = PScan.apply

def serial_scan(A, X):
    B = A.size(0)
    L = A.size(1)
    H = torch.zeros_like(X)
    
    A_flat = A.view(B, L, -1)
    X_flat = X.view(B, L, -1)
    H_flat = H.view(B, L, -1)
    
    for b in range(B):
        for l in range(L):
            if l == 0:
                H_flat[b, l] = X_flat[b, l].clone()
            else:
                H_flat[b, l] = A_flat[b, l] * H_flat[b, l - 1].clone() + X_flat[b, l].clone()
                
    return H

def pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
    if not torch.cuda.is_available():
        return True
    device = 'cuda'
    dtype = torch.complex64
    
    A = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    
    H_pscan = pscan(A, X)
    H_serial = serial_scan(A, X)
    
    fwd_diff = (H_pscan - H_serial).abs().max()
    if fwd_diff > 1e-4:
        return False
        
    loss = H_pscan.sum().abs()
    loss.backward()
    return True
