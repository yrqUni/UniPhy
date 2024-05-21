import os
import torch
import torch.nn as nn
from torchvision import models
class ResNetEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet_type = args.resnet_type
        resnet_path = args.resnet_path
        pretrained = args.resnet_pretrained
        trainable = args.resnet_trainable
        input_size = args.input_size
        self.output_ch = args.emb_ch
        if pretrained:
            try:
                resnet = getattr(models, resnet_type.lower())(pretrained=pretrained)
            except:
                resnet = getattr(models, resnet_type.lower())(pretrained=False)
                weight_path = os.path.join(resnet_path, f'{resnet_type}_pretrained.pth')
                resnet.load_state_dict(torch.load(weight_path))
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        if not trainable:
            for param in self.features.parameters():
                param.requires_grad = False
        self.upsample_in = nn.Upsample(scale_factor=args.resnet_scale_factor, mode='bilinear', align_corners=True)
        self.pre_conv = nn.Conv2d(args.input_ch, 3, kernel_size=3, padding='same')
        with torch.no_grad():
            sample_input = torch.zeros(1, args.input_ch, *input_size)
            sample_input = self.upsample_in(sample_input)
            sample_output = self.features(self.pre_conv(sample_input))
            output_size = (sample_output.size(2), sample_output.size(3))
        scale_factor = (input_size[0] / output_size[0], input_size[1] / output_size[1])
        self.upsample_out = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.downsp = nn.Conv2d(in_channels=resnet.inplanes, out_channels=self.output_ch, kernel_size=args.hidden_factor, stride=args.hidden_factor)
        with torch.no_grad():
            x = torch.zeros(1, resnet.inplanes, *input_size)
            x = self.downsp(x)
            _, C, H, W = x.size()
            self.input_downsp_shape = (C, H, W)
    def forward(self, x):
        B, L, C, H, W = x.size()
        x = x.view(B * L, C, H, W)
        x = self.upsample_in(x)
        x = self.pre_conv(x)  
        x = self.features(x)
        x = self.upsample_out(x)
        x = self.downsp(x)
        _, C, H, W = x.size()
        x = x.view(B, L, C, H, W)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        resnet_type = args.resnet_type
        resnet_path = args.resnet_path
        pretrained = args.resnet_pretrained
        trainable = args.resnet_trainable
        output_ch = args.input_ch
        if pretrained:
            try:
                resnet = getattr(models, resnet_type.lower())(pretrained=pretrained)
            except:
                resnet = getattr(models, resnet_type.lower())(pretrained=False)
                weight_path = os.path.join(resnet_path, f'{resnet_type}_pretrained.pth')
                resnet.load_state_dict(torch.load(weight_path))
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        if not trainable:
            for param in self.features.parameters():
                param.requires_grad = False
        self.upsample_in = nn.Upsample(scale_factor=args.resnet_scale_factor, mode='bilinear', align_corners=True)
        self.pre_conv = nn.Conv2d(args.emb_ch, 3, kernel_size=3, padding='same')  
        with torch.no_grad():
            sample_input = torch.zeros(1, args.emb_ch, *input_downsp_shape[1:])
            sample_input = self.upsample_in(sample_input)
            sample_output = self.features(self.pre_conv(sample_input))
            output_size = (sample_output.size(2), sample_output.size(3))
        scale_factor = (input_downsp_shape[1] / output_size[0], input_downsp_shape[2] / output_size[1])
        self.upsample_out = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.upsp = nn.ConvTranspose2d(in_channels=resnet.inplanes, out_channels=input_downsp_shape[0], kernel_size=args.hidden_factor, stride=args.hidden_factor)
        self.out_conv = nn.Conv2d(in_channels=input_downsp_shape[0], out_channels=output_ch, kernel_size=1, padding='same')
    def forward(self, x):
        B, L, C, H, W = x.size()
        x = x.view(B * L, C, H, W)
        x = self.upsample_in(x)
        x = self.pre_conv(x) 
        x = self.features(x)
        x = self.upsample_out(x)
        x = self.upsp(x)
        x = self.out_conv(x)
        _, C, H, W = x.size()
        x = x.view(B, L, C, H, W)
        return x

import torch
import torch.nn as nn

class Args:
    def __init__(self):
        self.resnet_type = 'resnet18'
        self.resnet_path = './resnet_ckpt'
        self.resnet_pretrained = True
        self.resnet_trainable = True
        self.input_size = (64, 32)
        self.input_ch = 3
        self.emb_ch = 256
        self.resnet_scale_factor = 16
        self.hidden_factor = (2,1)

args = Args()

embedding = ResNetEmbedding(args)
decoder = ResNetDecoder(args, embedding.input_downsp_shape)

B, L, C, H, W = 4, 10, 3, args.input_size[0], args.input_size[1]
test_input = torch.randn(B, L, C, H, W)

embedding_output = embedding(test_input)
decoder_output = decoder(embedding_output)

print(f"Input shape: {test_input.shape}")
print(f"Embedding output shape: {embedding_output.shape}")
print(f"Decoder output shape: {decoder_output.shape}")
