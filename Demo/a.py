import torch
import torch.nn as nn

# 定义你的 RMSNorm 类
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if not self.use_mup:
            return output * self.weight
        else:
            return output

# 测试函数
def test_rms_norm_complex():
    # 生成复数张量
    real_part = torch.randn(4, 10)
    imag_part = torch.randn(4, 10)
    complex_tensor = torch.complex(real_part, imag_part)

    # 实例化 RMSNorm
    rms_norm = RMSNorm(d_model=10)

    # 应用 RMSNorm 到复数张量
    output = rms_norm(complex_tensor)

    # 打印输出和类型
    print("Input Complex Tensor:")
    print(complex_tensor)
    print("Output Complex Tensor after RMSNorm:")
    print(output)
    print("Output dtype:", output.dtype)

# 运行测试
test_rms_norm_complex()
