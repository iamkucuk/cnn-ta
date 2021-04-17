from torch import nn
from torch.nn.modules import Conv2d, ReLU, MaxPool2d, Flatten, Dropout, Linear, Sequential
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(1, 32, (3, 3)),
            ReLU(),
            Conv2d(32, 64, (3, 3)),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Dropout(.25)
        )

        dummy_input = torch.zeros((1, 1, 14, 15))
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[-1]
        
        self.dropout = Dropout(.5)
        self.pred_layer = Linear(feature_size, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.pred_layer(x)