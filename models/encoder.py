import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        x = self.encoder(x)
        return x
