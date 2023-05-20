import torch
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplemented
