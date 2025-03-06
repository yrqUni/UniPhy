import torch.nn as nn

class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_size=3, stride=1, padding=1):
        super(Conv3D, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  
        self.conv3d_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        output = self.conv3d_layers(x)
        output = output.permute(0, 2, 1, 3, 4)
        return output

# class Args:
#     def __init__(self):
#         self.in_channels = 3   
#         self.out_channels = 8  
#         self.num_blocks = 4  

# if __name__ == "__main__":
#     args = Args()
#     model = Conv3DModel(
#         in_channels=args.in_channels,
#         out_channels=args.out_channels,
#         num_blocks=args.num_blocks
#     )

#     input_tensor = torch.randn(2, 5, args.in_channels, 64, 64)
#     output_tensor = model(input_tensor)

#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)
