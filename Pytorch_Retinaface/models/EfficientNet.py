import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)

    def forward(self, x):
        return self.backbone.extract_endpoints(x)