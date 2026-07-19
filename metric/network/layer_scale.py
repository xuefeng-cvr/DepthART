import torch
import torch.nn as nn
from typing import Union, Optional


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        
        # 处理不同类型的 init_values
        if isinstance(init_values, torch.Tensor):
            if init_values.numel() == 1:
                # 如果是标量张量，转换为数值
                init_val = init_values.item()
                self.gamma = nn.Parameter(init_val * torch.ones(dim))
            elif init_values.shape[0] == dim:
                # 如果已经是正确形状的张量
                self.gamma = nn.Parameter(init_values)
            else:
                raise ValueError(f"init_values tensor shape {init_values.shape} doesn't match dim {dim}")
        else:
            # 如果是数值类型
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma