import torch
import torch.nn as nn
import torch.optim as optim
from ConvLSTM import ConvLSTM
# 创建一个随机数据集
batch_size = 10
seq_length = 5
channels = 64
height = 32
width = 32
output_dim = 1  # 假设我们的输出是一个单一的预测值

# 随机生成输入和目标输出
input_tensor = torch.randn(batch_size, seq_length, channels, height, width)
target = torch.randn(batch_size, output_dim)

# 初始化ConvLSTM模型
model = ConvLSTM(input_dim=channels, hidden_dim=[128, 64], kernel_size=(3, 3), num_layers=2, batch_first=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 正向传播
    outputs, _ = model(input_tensor)
    # 只取最后一个时间步的输出用于预测
    outputs = outputs[-1][:, -1, :, :, :]
    outputs = outputs.mean(dim=[2, 3])  # 简单地对空间维度求平均以匹配目标尺寸

    # 计算损失
    loss = criterion(outputs, target)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 注意：这个测试例子非常简化，实际使用时需要更复杂的数据处理和更细致的模型配置。
