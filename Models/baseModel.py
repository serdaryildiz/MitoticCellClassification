from torch import nn


class BaseModel(nn.Module):
    """
        Base Model
    """
    def __init__(self, **kwargs):
        super().__init__()
        return

    def forward(self, x):
        raise NotImplemented
