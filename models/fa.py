import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregator(nn.Module):
    def __init__(self, in_channels_list):
        super(FeatureAggregator, self).__init__()
        self.upsample_layers = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=1) 
                                              for in_channels in in_channels_list])

    def forward(self, features):
        upsampled_features = [F.interpolate(upsample_layer(feature), size=features[0].shape[2:], mode='bilinear')
                              for upsample_layer, feature in zip(self.upsample_layers, features)]
        aggregated_features = torch.cat(upsampled_features, dim=1)
        return aggregated_features
