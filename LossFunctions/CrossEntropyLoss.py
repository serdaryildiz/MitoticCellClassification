import torch
from torch.nn import functional as F
from LossFunctions import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        return

    def forward(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(prediction, labels)
        return loss
