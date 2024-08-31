import torch
import torch.nn as nn

class ScaleInvariantBottleneck(nn.Module):
    def __init__(self, in_channels, num_layers=3):
        super(ScaleInvariantBottleneck, self).__init__()
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ) for _ in range(num_layers)])
        
        self.decoders = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ) for _ in range(num_layers)])
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for encoder, decoder in zip(self.encoders, self.decoders):
            encoded = encoder(x)
            decoded = decoder(encoded)
            x = x + decoded
        x = self.attention(x) * x
        return x
