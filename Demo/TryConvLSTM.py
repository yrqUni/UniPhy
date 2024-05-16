import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from model.ConvLSTM import ConvLSTM

batch_size = 10
seq_length = 5
channels = 64
height = 32
width = 32
output_dim = 1  

input_tensor = torch.randn(batch_size, seq_length, channels, height, width)
target = torch.randn(batch_size, output_dim)

model = ConvLSTM(input_dim=channels, hidden_dim=[128, 64], kernel_size=(3, 3), num_layers=2, batch_first=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs, _ = model(input_tensor)

    outputs = outputs[-1][:, -1, :, :, :]
    outputs = outputs.mean(dim=[2, 3])  

    loss = criterion(outputs, target)

    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
