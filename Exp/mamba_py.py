# import math

# import torch
# import torch.nn.functional as F

# """

# An implementation of the parallel scan operation in PyTorch (Blelloch version).
# Please see docs/pscan.ipynb for a detailed explanation of what happens here.

# """

# def npo2(len):
#     """
#     Returns the next power of 2 above len
#     """

#     return 2 ** math.ceil(math.log2(len))

# def pad_npo2(X):
#     """
#     Pads input length dim to the next power of 2

#     Args:
#         X : (B, L, D, N)

#     Returns:
#         Y : (B, npo2(L), D, N)
#     """

#     len_npo2 = npo2(X.size(1))
#     pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
#     return F.pad(X, pad_tuple, "constant", 0)

# class PScan(torch.autograd.Function):
#     @staticmethod
#     def pscan(A, X):
#         # A : (B, D, L, N)
#         # X : (B, D, L, N)

#         # modifies X in place by doing a parallel scan.
#         # more formally, X will be populated by these values :
#         # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
#         # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

#         # only supports L that is a power of two (mainly for a clearer code)
        
#         B, D, L, _ = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps-2):
#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)
            
#             Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
#             Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

#             Aa = Aa[:, :, :, 1]
#             Xa = Xa[:, :, :, 1]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(2) == 4:
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Aa[:, :, 1].mul_(Aa[:, :, 0])

#             Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
#         elif Xa.size(2) == 2:
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
#         Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
#         Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
#         Aa[:, :, 2].mul_(Aa[:, :, 1])

#         for k in range(num_steps-3, -1, -1):
#             Aa = A[:, :, 2**k-1:L:2**k]
#             Xa = X[:, :, 2**k-1:L:2**k]

#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)
            
#             Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
#             Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

#     @staticmethod
#     def pscan_rev(A, X):
#         # A : (B, D, L, N)
#         # X : (B, D, L, N)

#         # the same function as above, but in reverse
#         # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
#         # it is used in the backward pass

#         # only supports L that is a power of two (mainly for a clearer code)

#         B, D, L, _ = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps-2):
#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)
                    
#             Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
#             Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

#             Aa = Aa[:, :, :, 0]
#             Xa = Xa[:, :, :, 0]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(2) == 4:
#             Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
#             Aa[:, :, 2].mul_(Aa[:, :, 3])

#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
#         elif Xa.size(2) == 2:
#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, :, 0:L:2**(num_steps-2)]
#         Xa = X[:, :, 0:L:2**(num_steps-2)]
#         Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
#         Aa[:, :, 1].mul_(Aa[:, :, 2])

#         for k in range(num_steps-3, -1, -1):
#             Aa = A[:, :, 0:L:2**k]
#             Xa = X[:, :, 0:L:2**k]

#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)

#             Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
#             Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

#     @staticmethod
#     def forward(ctx, A_in, X_in):
#         """
#         Applies the parallel scan operation, as defined above. Returns a new tensor.
#         If you can, privilege sequence lengths that are powers of two.

#         Args:
#             A_in : (B, L, D, N)
#             X_in : (B, L, D, N)

#         Returns:
#             H : (B, L, D, N)
#         """

#         L = X_in.size(1)

#         # cloning is requiered because of the in-place ops
#         if L == npo2(L):
#             A = A_in.clone()
#             X = X_in.clone()
#         else:
#             # pad tensors (and clone btw)
#             A = pad_npo2(A_in) # (B, npo2(L), D, N)
#             X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
#         # prepare tensors
#         A = A.transpose(2, 1) # (B, D, npo2(L), N)
#         X = X.transpose(2, 1) # (B, D, npo2(L), N)

#         # parallel scan (modifies X in-place)
#         PScan.pscan(A, X)

#         ctx.save_for_backward(A_in, X)
        
#         # slice [:, :L] (cut if there was padding)
#         return X.transpose(2, 1)[:, :L]
    
#     @staticmethod
#     def backward(ctx, grad_output_in):
#         """
#         Flows the gradient from the output to the input. Returns two new tensors.

#         Args:
#             ctx : A_in : (B, L, D, N), X : (B, D, L, N)
#             grad_output_in : (B, L, D, N)

#         Returns:
#             gradA : (B, L, D, N), gradX : (B, L, D, N)
#         """

#         A_in, X = ctx.saved_tensors

#         L = grad_output_in.size(1)

#         # cloning is requiered because of the in-place ops
#         if L == npo2(L):
#             grad_output = grad_output_in.clone()
#             # the next padding will clone A_in
#         else:
#             grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
#             A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

#         # prepare tensors
#         grad_output = grad_output.transpose(2, 1)
#         A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
#         A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

#         # reverse parallel scan (modifies grad_output in-place)
#         PScan.pscan_rev(A, grad_output)

#         Q = torch.zeros_like(X)
#         Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

#         return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
# pscan = PScan.apply

# ################################################################################

# def serial_scan(A, X):
#     """
#     Serial implementation of the scan operation.
    
#     Args:
#         A : (B, L, D, N)
#         X : (B, L, D, N)
    
#     Returns:
#         H : (B, L, D, N)
#     """
#     B, L, D, N = A.size()
#     H = torch.zeros_like(X)
#     for b in range(B):
#         for d in range(D):
#             for l in range(L):
#                 if l == 0:
#                     H[b, l, d] = X[b, l, d]
#                 else:
#                     H[b, l, d] = A[b, l, d] * H[b, l-1, d] + X[b, l, d]
#     return H

# # Test for consistency
# B, L, D, N = 2, 8, 768, 32  # Batch size, sequence length, feature dimension, and additional dimension
# A = torch.rand(B, L, D, N)
# X = torch.rand(B, L, D, N)

# # Compute using parallel scan
# pscan_result = pscan(A, X)

# # Compute using serial scan
# serial_scan_result = serial_scan(A, X)

# # Check if the results are close
# print(torch.allclose(pscan_result, serial_scan_result))

# ################################################################################
# ################################################################################

# import math
# import torch
# import torch.nn.functional as F

# def npo2(len):
#     """
#     Returns the next power of 2 above len
#     """
#     return 2 ** math.ceil(math.log2(len))

# def pad_npo2(X):
#     """
#     Pads input length dim to the next power of 2

#     Args:
#         X : (B, L, D, N)

#     Returns:
#         Y : (B, npo2(L), D, N)
#     """
#     len_npo2 = npo2(X.size(1))
#     pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
#     return F.pad(X, pad_tuple, "constant", 0)

# class PScan(torch.autograd.Function):
#     @staticmethod
#     def pscan(A, X):
#         # A : (B, L, D, N)
#         # X : (B, L, D, N)

#         B, L, D, N = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, D, N)
#             Xa = Xa.view(B, T // 2, 2, D, N)
            
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Aa[:, :, 1].mul_(Aa[:, :, 0])

#             Aa = Aa[:, :, 1]
#             Xa = Xa[:, :, 1]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             Aa[:, 1].mul_(Aa[:, 0])

#             Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
#         elif Xa.size(1) == 2:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa = X[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 1]))
#         Aa[:, 2].mul_(Aa[:, 1])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 2 ** k - 1:L:2 ** k]
#             Xa = X[:, 2 ** k - 1:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, D, N)
#             Xa = Xa.view(B, T // 2, 2, D, N)
            
#             Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
#             Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

#     @staticmethod
#     def pscan_rev(A, X):
#         # A : (B, L, D, N)
#         # X : (B, L, D, N)

#         B, L, D, N = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, D, N)
#             Xa = Xa.view(B, T // 2, 2, D, N)
                    
#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
#             Aa[:, :, 0].mul_(Aa[:, :, 1])

#             Aa = Aa[:, :, 0]
#             Xa = Xa[:, :, 0]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
#             Aa[:, 2].mul_(Aa[:, 3])

#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
#         elif Xa.size(1) == 2:
#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 0:L:2 ** (num_steps - 2)]
#         Xa = X[:, 0:L:2 ** (num_steps - 2)]
#         Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 2]))
#         Aa[:, 1].mul_(Aa[:, 2])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 0:L:2 ** k]
#             Xa = X[:, 0:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, D, N)
#             Xa = Xa.view(B, T // 2, 2, D, N)

#             Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
#             Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

#     @staticmethod
#     def forward(ctx, A_in, X_in):
#         """
#         Applies the parallel scan operation, as defined above. Returns a new tensor.
#         If you can, privilege sequence lengths that are powers of two.

#         Args:
#             A_in : (B, L, D, N)
#             X_in : (B, L, D, N)

#         Returns:
#             H : (B, L, D, N)
#         """

#         L = X_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             A = A_in.clone()
#             X = X_in.clone()
#         else:
#             # pad tensors (and clone btw)
#             A = pad_npo2(A_in) # (B, npo2(L), D, N)
#             X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
#         # parallel scan (modifies X in-place)
#         PScan.pscan(A, X)

#         ctx.save_for_backward(A_in, X)
        
#         # slice [:, :L] (cut if there was padding)
#         return X[:, :L]
    
#     @staticmethod
#     def backward(ctx, grad_output_in):
#         """
#         Flows the gradient from the output to the input. Returns two new tensors.

#         Args:
#             ctx : A_in : (B, L, D, N), X : (B, D, L, N)
#             grad_output_in : (B, L, D, N)

#         Returns:
#             gradA : (B, L, D, N), gradX : (B, L, D, N)
#         """

#         A_in, X = ctx.saved_tensors

#         L = grad_output_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             grad_output = grad_output_in.clone()
#             # the next padding will clone A_in
#         else:
#             grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
#             A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

#         A = A_in
#         # shift 1 to the left (see hand derivation)
#         A_shifted = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 1))

#         # reverse parallel scan (modifies grad_output in-place)
#         PScan.pscan_rev(A_shifted, grad_output)

#         Q = torch.zeros_like(X)
#         Q[:, 1:].add_(X[:, :-1] * grad_output[:, 1:])

#         return Q[:, :L], grad_output[:, :L]
    
# pscan = PScan.apply

# ################################################################################

# def serial_scan(A, X):
#     """
#     Serial implementation of the scan operation.
    
#     Args:
#         A : (B, L, D, N)
#         X : (B, L, D, N)
    
#     Returns:
#         H : (B, L, D, N)
#     """
#     B, L, D, N = A.size()
#     H = torch.zeros_like(X)
#     for b in range(B):
#         for d in range(D):
#             for l in range(L):
#                 if l == 0:
#                     H[b, l, d] = X[b, l, d]
#                 else:
#                     H[b, l, d] = A[b, l, d] * H[b, l - 1, d] + X[b, l, d]
#     return H

# # Test for consistency
# B, L, D, N = 2, 8, 768, 32  # Batch size, sequence length, feature dimension, and additional dimension
# A = torch.rand(B, L, D, N)
# X = torch.rand(B, L, D, N)

# # Compute using parallel scan
# pscan_result = pscan(A, X)

# # Compute using serial scan
# serial_scan_result = serial_scan(A, X)

# # Check if the results are close
# print(torch.allclose(pscan_result, serial_scan_result))

# ################################################################################
# ################################################################################

# import math
# import torch
# import torch.nn.functional as F

# def npo2(len):
#     """
#     Returns the next power of 2 above len
#     """
#     return 2 ** math.ceil(math.log2(len))

# def pad_npo2(X):
#     """
#     Pads input length dim to the next power of 2

#     Args:
#         X : (B, L, C, H, W)

#     Returns:
#         Y : (B, npo2(L), C, H, W)
#     """
#     len_npo2 = npo2(X.size(1))
#     pad_tuple = (0, 0, 0, 0, 0, 0, 0, len_npo2 - X.size(1))
#     return F.pad(X, pad_tuple, "constant", 0)

# class PScan(torch.autograd.Function):
#     @staticmethod
#     def pscan(A, X):
#         # A : (B, L, C, H, W)
#         # X : (B, L, C, H, W)

#         B, L, C, H, W = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, H, W)
#             Xa = Xa.view(B, T // 2, 2, C, H, W)
            
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Aa[:, :, 1].mul_(Aa[:, :, 0])

#             Aa = Aa[:, :, 1]
#             Xa = Xa[:, :, 1]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             Aa[:, 1].mul_(Aa[:, 0])

#             Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
#         elif Xa.size(1) == 2:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa = X[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 1]))
#         Aa[:, 2].mul_(Aa[:, 1])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 2 ** k - 1:L:2 ** k]
#             Xa = X[:, 2 ** k - 1:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, H, W)
#             Xa = Xa.view(B, T // 2, 2, C, H, W)
            
#             Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
#             Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

#     @staticmethod
#     def pscan_rev(A, X):
#         # A : (B, L, C, H, W)
#         # X : (B, L, C, H, W)

#         B, L, C, H, W = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, H, W)
#             Xa = Xa.view(B, T // 2, 2, C, H, W)
                    
#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
#             Aa[:, :, 0].mul_(Aa[:, :, 1])

#             Aa = Aa[:, :, 0]
#             Xa = Xa[:, :, 0]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
#             Aa[:, 2].mul_(Aa[:, 3])

#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
#         elif Xa.size(1) == 2:
#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 0:L:2 ** (num_steps - 2)]
#         Xa = X[:, 0:L:2 ** (num_steps - 2)]
#         Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 2]))
#         Aa[:, 1].mul_(Aa[:, 2])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 0:L:2 ** k]
#             Xa = X[:, 0:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, H, W)
#             Xa = Xa.view(B, T // 2, 2, C, H, W)

#             Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
#             Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

#     @staticmethod
#     def forward(ctx, A_in, X_in):
#         """
#         Applies the parallel scan operation, as defined above. Returns a new tensor.
#         If you can, privilege sequence lengths that are powers of two.

#         Args:
#             A_in : (B, L, C, H, W)
#             X_in : (B, L, C, H, W)

#         Returns:
#             H : (B, L, C, H, W)
#         """

#         L = X_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             A = A_in.clone()
#             X = X_in.clone()
#         else:
#             # pad tensors (and clone btw)
#             A = pad_npo2(A_in) # (B, npo2(L), C, H, W)
#             X = pad_npo2(X_in) # (B, npo2(L), C, H, W)
        
#         # parallel scan (modifies X in-place)
#         PScan.pscan(A, X)

#         ctx.save_for_backward(A_in, X)
        
#         # slice [:, :L] (cut if there was padding)
#         return X[:, :L]
    
#     @staticmethod
#     def backward(ctx, grad_output_in):
#         """
#         Flows the gradient from the output to the input. Returns two new tensors.

#         Args:
#             ctx : A_in : (B, L, C, H, W), X : (B, L, C, H, W)
#             grad_output_in : (B, L, C, H, W)

#         Returns:
#             gradA : (B, L, C, H, W), gradX : (B, L, C, H, W)
#         """

#         A_in, X = ctx.saved_tensors

#         L = grad_output_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             grad_output = grad_output_in.clone()
#             # the next padding will clone A_in
#         else:
#             grad_output = pad_npo2(grad_output_in) # (B, npo2(L), C, H, W)
#             A_in = pad_npo2(A_in) # (B, npo2(L), C, H, W)

#         A = A_in
#         # shift 1 to the left (see hand derivation)
#         A_shifted = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 0, 0, 1))

#         # reverse parallel scan (modifies grad_output in-place)
#         PScan.pscan_rev(A_shifted, grad_output)

#         Q = torch.zeros_like(X)
#         Q[:, 1:].add_(X[:, :-1] * grad_output[:, 1:])

#         return Q[:, :L], grad_output[:, :L]
    
# pscan = PScan.apply

# ################################################################################

# def serial_scan(A, X):
#     """
#     Serial implementation of the scan operation.
    
#     Args:
#         A : (B, L, C, H, W)
#         X : (B, L, C, H, W)
    
#     Returns:
#         H : (B, L, C, H, W)
#     """
#     B, L, C, H, W = A.size()
#     H = torch.zeros_like(X)
#     for b in range(B):
#         for c in range(C):
#             for l in range(L):
#                 if l == 0:
#                     H[b, l, c] = X[b, l, c]
#                 else:
#                     H[b, l, c] = A[b, l, c] * H[b, l - 1, c] + X[b, l, c]
#     return H

# # Test for consistency
# B, L, C, H, W = 2, 8, 768, 32, 32  # Batch size, sequence length, channels, and spatial dimensions
# A = torch.rand(B, L, C, H, W)
# X = torch.rand(B, L, C, H, W)

# # Compute using parallel scan
# pscan_result = pscan(A, X)

# # Compute using serial scan
# serial_scan_result = serial_scan(A, X)

# # Check if the results are close
# print(torch.allclose(pscan_result, serial_scan_result))

# ################################################################################
# ################################################################################

# import math
# import torch
# import torch.nn.functional as F

# def npo2(len):
#     """
#     Returns the next power of 2 above len
#     """
#     return 2 ** math.ceil(math.log2(len))

# def pad_npo2(X):
#     """
#     Pads input length dim to the next power of 2

#     Args:
#         X : (B, L, C, S, S)

#     Returns:
#         Y : (B, npo2(L), C, S, S)
#     """
#     len_npo2 = npo2(X.size(1))
#     pad_tuple = (0, 0, 0, 0, 0, 0, 0, len_npo2 - X.size(1))
#     return F.pad(X, pad_tuple, "constant", 0)

# class PScan(torch.autograd.Function):
#     @staticmethod
#     def pscan(A, X):
#         # A : (B, L, C, S, S)
#         # X : (B, L, C, S, S)

#         B, L, C, S1, S2 = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, S1, S2)
#             Xa = Xa.view(B, T // 2, 2, C, S1, S2)
            
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Aa[:, :, 1].mul_(Aa[:, :, 0])

#             Aa = Aa[:, :, 1]
#             Xa = Xa[:, :, 1]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             Aa[:, 1].mul_(Aa[:, 0])

#             Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
#         elif Xa.size(1) == 2:
#             Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa = X[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
#         Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 1]))
#         Aa[:, 2].mul_(Aa[:, 1])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 2 ** k - 1:L:2 ** k]
#             Xa = X[:, 2 ** k - 1:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, S1, S2)
#             Xa = Xa.view(B, T // 2, 2, C, S1, S2)
            
#             Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
#             Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

#     @staticmethod
#     def pscan_rev(A, X):
#         # A : (B, L, C, S, S)
#         # X : (B, L, C, S, S)

#         B, L, C, S1, S2 = A.size()
#         num_steps = int(math.log2(L))

#         # up sweep (last 2 steps unfolded)
#         Aa = A
#         Xa = X
#         for _ in range(num_steps - 2):
#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, S1, S2)
#             Xa = Xa.view(B, T // 2, 2, C, S1, S2)
                    
#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
#             Aa[:, :, 0].mul_(Aa[:, :, 1])

#             Aa = Aa[:, :, 0]
#             Xa = Xa[:, :, 0]

#         # we have only 4, 2 or 1 nodes left
#         if Xa.size(1) == 4:
#             Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
#             Aa[:, 2].mul_(Aa[:, 3])

#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
#         elif Xa.size(1) == 2:
#             Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
#             return
#         else:
#             return

#         # down sweep (first 2 steps unfolded)
#         Aa = A[:, 0:L:2 ** (num_steps - 2)]
#         Xa = X[:, 0:L:2 ** (num_steps - 2)]
#         Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 2]))
#         Aa[:, 1].mul_(Aa[:, 2])

#         for k in range(num_steps - 3, -1, -1):
#             Aa = A[:, 0:L:2 ** k]
#             Xa = X[:, 0:L:2 ** k]

#             T = Xa.size(1)
#             Aa = Aa.view(B, T // 2, 2, C, S1, S2)
#             Xa = Xa.view(B, T // 2, 2, C, S1, S2)

#             Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
#             Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

#     @staticmethod
#     def forward(ctx, A_in, X_in):
#         """
#         Applies the parallel scan operation, as defined above. Returns a new tensor.
#         If you can, privilege sequence lengths that are powers of two.

#         Args:
#             A_in : (B, L, C, S, S)
#             X_in : (B, L, C, S, S)

#         Returns:
#             H : (B, L, C, S, S)
#         """

#         L = X_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             A = A_in.clone()
#             X = X_in.clone()
#         else:
#             # pad tensors (and clone btw)
#             A = pad_npo2(A_in) # (B, npo2(L), C, S, S)
#             X = pad_npo2(X_in) # (B, npo2(L), C, S, S)
        
#         # parallel scan (modifies X in-place)
#         PScan.pscan(A, X)

#         ctx.save_for_backward(A_in, X)
        
#         # slice [:, :L] (cut if there was padding)
#         return X[:, :L]
    
#     @staticmethod
#     def backward(ctx, grad_output_in):
#         """
#         Flows the gradient from the output to the input. Returns two new tensors.

#         Args:
#             ctx : A_in : (B, L, C, S, S), X : (B, L, C, S, S)
#             grad_output_in : (B, L, C, S, S)

#         Returns:
#             gradA : (B, L, C, S, S), gradX : (B, L, C, S, S)
#         """

#         A_in, X = ctx.saved_tensors

#         L = grad_output_in.size(1)

#         # cloning is required because of the in-place ops
#         if L == npo2(L):
#             grad_output = grad_output_in.clone()
#             # the next padding will clone A_in
#         else:
#             grad_output = pad_npo2(grad_output_in) # (B, npo2(L), C, S, S)
#             A_in = pad_npo2(A_in) # (B, npo2(L), C, S, S)

#         A = A_in
#         # shift 1 to the left (see hand derivation)
#         A_shifted = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 0, 0, 1))

#         # reverse parallel scan (modifies grad_output in-place)
#         PScan.pscan_rev(A_shifted, grad_output)

#         Q = torch.zeros_like(X)
#         Q[:, 1:].add_(X[:, :-1] * grad_output[:, 1:])

#         return Q[:, :L], grad_output[:, :L]
    
# pscan = PScan.apply

# ################################################################################

# def serial_scan(A, X):
#     """
#     Serial implementation of the scan operation.
    
#     Args:
#         A : (B, L, C, S, S)
#         X : (B, L, C, S, S)
    
#     Returns:
#         H : (B, L, C, S, S)
#     """
#     B, L, C, S1, S2 = A.size()
#     H = torch.zeros_like(X)
#     for b in range(B):
#         for c in range(C):
#             for l in range(L):
#                 if l == 0:
#                     H[b, l, c] = X[b, l, c]
#                 else:
#                     H[b, l, c] = A[b, l, c] * H[b, l - 1, c] + X[b, l, c]
#     return H

# # Test for consistency
# B, L, C, S1, S2 = 2, 8, 768, 32, 44  # Batch size, sequence length, channels, and spatial dimensions
# A = torch.rand(B, L, C, S1, S2)
# X = torch.rand(B, L, C, S1, S2)

# # Compute using parallel scan
# pscan_result = pscan(A, X)

# # Compute using serial scan
# serial_scan_result = serial_scan(A, X)

# # Check if the results are close
# print(torch.allclose(pscan_result, serial_scan_result))

################################################################################
################################################################################

import math
import torch
import torch.nn.functional as F

def npo2(len):
    """
    Returns the next power of 2 above len
    """
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

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
        # A : (B, L, C, S, 1)
        # X : (B, L, C, S, S)

        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
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

        # we have only 4, 2 or 1 nodes left
        if Xa.size(1) == 4:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            Aa[:, 1].mul_(Aa[:, 0])

            Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
        elif Xa.size(1) == 2:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
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
        # A : (B, L, C, S, 1)
        # X : (B, L, C, S, S)

        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
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

        # we have only 4, 2 or 1 nodes left
        if Xa.size(1) == 4:
            Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
            Aa[:, 2].mul_(Aa[:, 3])

            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
        elif Xa.size(1) == 2:
            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
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
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, C, S, 1)
            X_in : (B, L, C, S, S)

        Returns:
            H : (B, L, C, S, S)
        """

        L = X_in.size(1)

        # cloning is required because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in) # (B, npo2(L), C, S, S)
            X = pad_npo2(X_in) # (B, npo2(L), C, S, S)
        
        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, C, S, 1), X : (B, L, C, S, S)
            grad_output_in : (B, L, C, S, S)

        Returns:
            gradA : (B, L, C, S, S), gradX : (B, L, C, S, S)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is required because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), C, S, S)
            A_in = pad_npo2(A_in) # (B, npo2(L), C, S, 1)

        A = A_in
        # shift 1 to the left (see hand derivation)
        A_shifted = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A_shifted, grad_output)

        Q = torch.zeros_like(X)
        Q[:, 1:].add_(X[:, :-1] * grad_output[:, 1:])

        return Q[:, :L], grad_output[:, :L]
    
pscan = PScan.apply

################################################################################
# Check correctness
###############################################################################
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
                H[b, l] = X[b, l]
            else:
                H[b, l] = A[b, l] * H[b, l - 1] + X[b, l]
    return H

B, L, C, S = 2, 8, 768, 32 
A = torch.nn.Parameter(torch.rand(1, 1, C, S, 1))
X = torch.rand(B, L, C, S, S)

pscan_result_p = pscan(A.expand(B, L, C, S, 1), X)
# pscan_result_i = A.expand(B, 2, C, S, 1) * pscan_result_p[:,-2].unsqueeze(1) + X[:, -1:]
pscan_result_i = pscan(A.expand(B, 2, C, S, 1), torch.concat([pscan_result_p[:,-2].unsqueeze(1).clone(),X[:, -1:].clone()], dim=1))
serial_scan_result = serial_scan(A.expand(B, L, C, S, 1), X)

print(torch.allclose(pscan_result_p, serial_scan_result))
print(torch.allclose(pscan_result_p[:, -1:], pscan_result_i[:, -1:]))
