import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell.

        Parameters:
        ----------
        input_dim: int
            Number of channels in the input tensor.
        hidden_dim: int
            Number of channels in the hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # Calculate the padding size to keep the input and output dimensions consistent
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Define the convolutional layer, input is input_dim + hidden_dim, output is 4 * hidden_dim (for i, f, o, g)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        # Split the output into four parts, corresponding to the input gate, forget gate, output gate, and candidate cell state
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update cell state
        c_next = f * c_cur + i * g
        # Update hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # Initialize hidden state and cell state with zeros
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM layer.

    Parameters:
    ----------
    input_dim: Number of input channels
    hidden_dim: Number of hidden channels
    kernel_size: Size of the convolutional kernel
    num_layers: Number of LSTM layers
    batch_first: Whether batch is the first dimension
    bias: Whether to use bias in convolutions
    return_all_layers: Whether to return the output of all layers

    Input:
    ------
    A tensor of shape B, T, C, H, W or T, B, C, H, W

    Output:
    ------
    A tuple containing two lists (length num_layers or length 1 if return_all_layers is False):
    0 - layer_output_list is a list of length T containing the output of each time step
    1 - last_state_list is the last state list, where each element is a (h, c) tuple of hidden state and cell state

    Example:
    >>> x = torch.rand((32, 10, 64, 128, 128))
    >>> convlstm = ConvLSTM(64, 16, (3, 3), 1, True, True, False)
    >>> _, last_states = convlstm(x)
    >>> h = last_states[0][0]  # 0 is the layer index, 0 is the h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # Check kernel_size consistency
        self._check_kernel_size_consistency(kernel_size)

        # Ensure kernel_size and hidden_dim lengths are consistent with num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create a list of ConvLSTMCells
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass function.

        Parameters:
        ----------
        input_tensor: Input tensor, shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: Initial hidden state, default is None

        Returns:
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # Change the order of input tensor if batch_first is False
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement the stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Initialize hidden state
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # Update state at each time step
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # Stack the outputs
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            # If not returning all layers, only return the last layer's output and state
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            # Initialize the hidden state for each layer
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# def test_convlstm():
#     # Set test parameters
#     B, T, C, H, W = 3, 10, 3, 64, 64  # Batch size, Time steps, Channels, Height, Width
#     hidden_dim = [16, 16, 16, 3]  # Hidden layer dimensions list
#     kernel_size = (3, 3)   # Convolutional kernel size
#     num_layers = len(hidden_dim)  # Number of layers
#     batch_first = True  # Whether batch is the first dimension
#     # Initialize ConvLSTM model
#     model = ConvLSTM(input_dim=C, hidden_dim=hidden_dim, kernel_size=kernel_size,
#                      num_layers=num_layers, batch_first=batch_first, bias=True, return_all_layers=False)

#     # Create input tensor
#     input_tensor = torch.randn(B, T, C, H, W)  # Randomly generate input data

#     # Run the model
#     layer_output_list, last_state_list = model(input_tensor)
    
#     # Print the output shape
#     print("Output shape: ", len(layer_output_list), len(layer_output_list[0]), layer_output_list[0][0].shape)

# # Run the test
# test_convlstm()
