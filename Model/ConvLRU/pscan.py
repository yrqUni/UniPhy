import math
import torch
import torch.nn.functional as F

def npo2(sequence_length):
    """Returns smallest power of 2 >= sequence_length"""
    return 2 ** math.ceil(math.log2(sequence_length))

def pnpo2(input_tensor):
    """Pads tensor's length dim to next power of 2"""
    target_length = npo2(input_tensor.size(1))
    pad_tuple = (0, 0, 0, 0, 0, 0, 0, target_length - input_tensor.size(1))
    return F.pad(input_tensor, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    """
    Parallel Scan algorithm implementation.
    
    Efficient O(log n) algorithm for recurrence H_t = A_t * H_{t-1} + X_t,
    versus O(n) for sequential computation.
    """
    
    @staticmethod
    def pscan(A, X):
        """
        Core parallel scan forward implementation.
        
        Args:
            A: Coefficient tensor (batch_size, seq_length, channels, state_dim, 1)
            X: Input tensor (batch_size, seq_length, channels, state_dim, state_dim)
        """
        batch_size, seq_length, channels, state_dim, _ = A.size()
        num_steps = int(math.log2(seq_length))

        # Initialize working tensors
        Aa = A
        Xa = X
        
        # Up-sweep phase
        for _ in range(num_steps - 2):
            current_length = Xa.size(1)
            Aa = Aa.view(batch_size, current_length // 2, 2, channels, state_dim, 1)
            Xa = Xa.view(batch_size, current_length // 2, 2, channels, state_dim, state_dim)
            
            product = Aa[:, :, 1].clone() * Xa[:, :, 0].clone()
            Xa[:, :, 1] = Xa[:, :, 1].clone() + product
            Aa[:, :, 1] = Aa[:, :, 1].clone() * Aa[:, :, 0].clone()

            Aa = Aa[:, :, 1]
            Xa = Xa[:, :, 1]

        # Special handling for length 4 case
        if Xa.size(1) == 4:
            product1 = Aa[:, 1].clone() * Xa[:, 0].clone()
            Xa[:, 1] = Xa[:, 1].clone() + product1
            Aa[:, 1] = Aa[:, 1].clone() * Aa[:, 0].clone()

            product2 = Aa[:, 2].clone() * Xa[:, 1].clone()
            product3 = Aa[:, 3].clone() * (Xa[:, 2].clone() + product2)
            Xa[:, 3] = Xa[:, 3].clone() + product3
        # Special handling for length 2 case
        elif Xa.size(1) == 2:
            product = Aa[:, 1].clone() * Xa[:, 0].clone()
            Xa[:, 1] = Xa[:, 1].clone() + product
            return
        else:
            return

        # Down-sweep phase
        Aa = A[:, 2 ** (num_steps - 2) - 1:seq_length:2 ** (num_steps - 2)]
        Xa = X[:, 2 ** (num_steps - 2) - 1:seq_length:2 ** (num_steps - 2)]
        
        product = Aa[:, 2].clone() * Xa[:, 1].clone()
        Xa[:, 2] = Xa[:, 2].clone() + product
        Aa[:, 2] = Aa[:, 2].clone() * Aa[:, 1].clone()

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 2 ** k - 1:seq_length:2 ** k]
            Xa = X[:, 2 ** k - 1:seq_length:2 ** k]

            current_length = Xa.size(1)
            Aa = Aa.view(batch_size, current_length // 2, 2, channels, state_dim, 1)
            Xa = Xa.view(batch_size, current_length // 2, 2, channels, state_dim, state_dim)
            
            product = Aa[:, 1:, 0].clone() * Xa[:, :-1, 1].clone()
            Xa[:, 1:, 0] = Xa[:, 1:, 0].clone() + product
            Aa[:, 1:, 0] = Aa[:, 1:, 0].clone() * Aa[:, :-1, 1].clone()

    @staticmethod
    def pscan_rev(A, X):
        """
        Reverse version of parallel scan (for backpropagation).
        
        Args:
            A: Coefficient tensor (batch_size, seq_length, channels, state_dim, 1)
            X: Input tensor (batch_size, seq_length, channels, state_dim, state_dim)
        """
        batch_size, seq_length, channels, state_dim, _ = A.size()
        num_steps = int(math.log2(seq_length))

        Aa = A
        Xa = X
        
        # Up-sweep phase (reverse)
        for _ in range(num_steps - 2):
            current_length = Xa.size(1)
            Aa = Aa.view(batch_size, current_length // 2, 2, channels, state_dim, 1)
            Xa = Xa.view(batch_size, current_length // 2, 2, channels, state_dim, state_dim)
            
            product = Aa[:, :, 0].clone() * Xa[:, :, 1].clone()
            Xa[:, :, 0] = Xa[:, :, 0].clone() + product
            Aa[:, :, 0] = Aa[:, :, 0].clone() * Aa[:, :, 1].clone()

            Aa = Aa[:, :, 0]
            Xa = Xa[:, :, 0]

        # Special handling for length 4 case
        if Xa.size(1) == 4:
            product1 = Aa[:, 2].clone() * Xa[:, 3].clone()
            Xa[:, 2] = Xa[:, 2].clone() + product1
            Aa[:, 2] = Aa[:, 2].clone() * Aa[:, 3].clone()

            product2 = Aa[:, 1].clone() * Xa[:, 2].clone()
            product3 = Aa[:, 0].clone() * (Xa[:, 1].clone() + product2)
            Xa[:, 0] = Xa[:, 0].clone() + product3
        # Special handling for length 2 case
        elif Xa.size(1) == 2:
            product = Aa[:, 0].clone() * Xa[:, 1].clone()
            Xa[:, 0] = Xa[:, 0].clone() + product
            return
        else:
            return

        # Down-sweep phase (reverse)
        Aa = A[:, 0:seq_length:2 ** (num_steps - 2)]
        Xa = X[:, 0:seq_length:2 ** (num_steps - 2)]
        
        product = Aa[:, 1].clone() * Xa[:, 2].clone()
        Xa[:, 1] = Xa[:, 1].clone() + product
        Aa[:, 1] = Aa[:, 1].clone() * Aa[:, 2].clone()

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 0:seq_length:2 ** k]
            Xa = X[:, 0:seq_length:2 ** k]

            current_length = Xa.size(1)
            Aa = Aa.view(batch_size, current_length // 2, 2, channels, state_dim, 1)
            Xa = Xa.view(batch_size, current_length // 2, 2, channels, state_dim, state_dim)
            
            product = Aa[:, :-1, 1].clone() * Xa[:, 1:, 0].clone()
            Xa[:, :-1, 1] = Xa[:, :-1, 1].clone() + product
            Aa[:, :-1, 1] = Aa[:, :-1, 1].clone() * Aa[:, 1:, 0].clone()

    @staticmethod
    def forward(ctx, A_in, X):
        """
        Forward pass function for parallel scan.
        
        Args:
            ctx: Context object for backward pass
            A_in: Coefficient tensor (batch_size, seq_length, channels, state_dim, 1)
            X: Input tensor (batch_size, seq_length, channels, state_dim, state_dim)
            
        Returns:
            Parallel scan result with original sequence length
        """
        original_length = X.size(1)
        
        # Check if padding is needed
        if original_length == npo2(original_length):
            A = A_in.clone()
            X_work = X.clone()
        else:
            A = pnpo2(A_in)
            X_work = pnpo2(X)
        
        PScan.pscan(A, X_work)
        ctx.save_for_backward(A_in, X_work)
        
        return X_work[:, :original_length]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Backward pass function computing gradients for A and X.
        
        Args:
            ctx: Context from forward pass
            grad_output_in: Output gradient
            
        Returns:
            Gradients for A_in and X
        """
        A, X = ctx.saved_tensors
        original_length = grad_output_in.size(1)
        
        # Handle padding for gradients
        if original_length == npo2(original_length):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pnpo2(grad_output_in)
            A = pnpo2(A)

        # Prepare for reverse scan
        A_padded = torch.nn.functional.pad(A[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))
        PScan.pscan_rev(A_padded, grad_output)

        # Compute gradient for A
        Q = torch.zeros_like(X)
        for i in range(1, X.size(1)):
            Q[:, i] = X[:, i-1].clone() * grad_output[:, i].clone()

        return Q[:, :original_length], grad_output[:, :original_length]

# Create function interface
pscan = PScan.apply

def serial_scan(A, X):
    """
    Serial implementation for verification purposes.
    
    Args:
        A: Coefficient tensor
        X: Input tensor
        
    Returns:
        Sequential scan result
    """
    batch_size, seq_length, channels, state_dim, _ = A.size()
    H = torch.zeros_like(X)
    
    H[:, 0] = X[:, 0].clone()
    for idx in range(1, seq_length):
        H[:, idx] = A[:, idx] * H[:, idx - 1].clone() + X[:, idx].clone()
    
    return H

def pscan_check(batch_size=2, seq_length=13, channels=8, state_dim=16):
    """
    Verifies the correctness of parallel scan implementation.
    
    Returns:
        (forward_match, gradient_match) tuple of booleans
    """
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
    
    return torch.allclose(H_pscan, H_serial_scan), torch.allclose(A1.grad, A2.grad)

# print(pscan_check())
