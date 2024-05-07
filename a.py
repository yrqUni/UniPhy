# import torch
# import torch.fft

# def global_convolution_via_fft(x, k):
#     """
#     使用傅立叶变换实现的全局卷积。
#     :param x: 输入张量，尺寸为 (batch_size, channels, height, width)
#     :param k: 卷积核张量，尺寸为 (channels, height, width)
#     :return: 卷积结果，尺寸为 (batch_size, channels, height, width)
#     """
#     # 扩展卷积核尺寸以匹配输入的批次和通道数
#     batch_size, channels, height, width = x.shape
#     k_expanded = k.unsqueeze(0).repeat(batch_size, 1, 1, 1)

#     # 对输入和卷积核进行傅立叶变换
#     X_fft = torch.fft.fft2(torch.fft.fft2(x))
#     K_fft = torch.fft.fft2(k_expanded)

#     # 在频域进行元素级乘法
#     Y_fft = X_fft * K_fft

#     # 进行逆傅立叶变换并取实部作为输出
#     y = torch.fft.ifft2(Y_fft).real

#     return y

# # 测试代码
# # 创建数据：batch_size=2, channels=3, height=width=64
# x = torch.rand(2, 3, 64, 64)
# k = torch.rand(3, 64, 64)

# # 调用函数
# output = global_convolution_via_fft(x, k)
# print(output.shape)

import torch

# 定义矩阵的形状
C, S = 3, 4  # 举例

# 随机生成矩阵 A 和 Z
A = torch.randn(C, S, S)
Z = torch.randn(C, S)

# 扩展 Z 的维度以匹配 A
Z_expanded = Z.unsqueeze(-1)

# 计算 Z*A
result_1 = Z_expanded * A

# 直接乘以对角矩阵 B
B = torch.diag_embed(Z)
print(B)
result_2 = torch.matmul(B, A)

# 验证两种方法的计算结果是否相等
assert torch.allclose(result_1, result_2), "结果不匹配"
print(result_1.shape)
print("结果匹配，验证通过！")
