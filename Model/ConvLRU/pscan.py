import math
import torch
import torch.nn.functional as F

def npo2(len):
    """
    Returns the next power of 2 above len.
    """
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2.
    Args:
        X : (B, L, C, S, S)
    Returns:
        Y : (B, npo2(L), C, S, S)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        """
        A : (B, L, C, S, 1)
        X : (B, L, C, S, S)
        """
        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            
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
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            
            Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        """
        A : (B, L, C, S, 1)
        X : (B, L, C, S, S)
        """
        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
                    
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
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)

            Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
            Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X):
        """
        Applies the parallel scan operation, as defined above. 
        Returns a new tensor.
        Privilege sequence lengths that are powers of two.
        Args:
            A_in : (B, L, C, S, 1)
            X : (B, L, C, S, S)
        Returns:
            H : (B, L, C, S, S)
        """
        L = X.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X.clone()
        else:
            A = pad_npo2(A_in) # (B, npo2(L), C, S, S)
            X = pad_npo2(X) # (B, npo2(L), C, S, S)
        
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        return X[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. 
        Returns two new tensors.
        Args:
            ctx : A : (B, L, C, S, 1), X : (B, L, C, S, S)
            grad_output_in : (B, L, C, S, S)
        Returns:
            gradA : (B, L, C, S, S), gradX : (B, L, C, S, S)
        """
        A, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), C, S, S)
            A = pad_npo2(A) # (B, npo2(L), C, S, 1)

        A = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))

        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, 1:].add_(X[:, :-1] * grad_output[:, 1:])

        return Q[:, :L], grad_output[:, :L]
    
pscan = PScan.apply

def serial_scan(A, X):
    """
    Serial implementation of the scan operation.
    Args:
        A : (B, L, C, S, 1)
        X : (B, L, C, S, S)
    Returns:
        H : (B, L, C, S, S)
    """
    B, L, C, S, S = A.size()
    H = torch.zeros_like(X)
    for b in range(B):
        for l in range(L):
            if l == 0:
                H[b, l] = X[b, l].clone()
            else:
                H[b, l] = A[b, l] * H[b, l - 1].clone() + X[b, l].clone()
    return H

def pscan_check(batch_size=2, seq_length=13, channels=8, state_dim=16):
    """
    Verifies the correctness of parallel scan implementation.
    
    Returns:
        (forward_match, gradient_match) tuple of booleans
    """
    pscan = PScan.apply
    # Generate random test data
    A_tensor = torch.rand(batch_size, seq_length, channels, state_dim, 1)
    A1 = torch.nn.Parameter(A_tensor.clone())
    A2 = torch.nn.Parameter(A_tensor.clone())
    X1 = torch.rand(batch_size, seq_length, channels, state_dim, state_dim)
    X2 = X1.clone()
    H_gt = torch.rand(batch_size, seq_length, channels, state_dim, state_dim)

    torch.autograd.set_detect_anomaly(True)
    loss_fn = torch.nn.MSELoss()
    
    # Test parallel implementation
    H_pscan = pscan(A1.expand(batch_size, seq_length, channels, state_dim, 1), X1)
    loss_pscan = loss_fn(H_pscan, H_gt)
    loss_pscan.backward()
    
    # Test serial implementation
    H_serial_scan = serial_scan(A2.expand(batch_size, seq_length, channels, state_dim, 1), X2)
    loss_serial_scan = loss_fn(H_serial_scan, H_gt)
    loss_serial_scan.backward()
    
    # Check if outputs and gradients match
    result = (torch.allclose(H_pscan, H_serial_scan), torch.allclose(A1.grad, A2.grad))
    
    # Clean up
    import gc
    del A_tensor, A1, A2, X1, X2, H_gt, H_pscan, H_serial_scan, loss_pscan, loss_serial_scan, loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    return result

# Run the test
# assert all(pscan_check()), "PScan implementation failed the test."
# print("PScan implementation passed the test.")
