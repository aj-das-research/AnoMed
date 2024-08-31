import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.class_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features):
        bbox_out = self.bbox_pred(features)
        class_out = self.class_pred(features)
        obj_out = self.obj_pred(features)
        return bbox_out, class_out, obj_out
    
    def compute_loss(self, bbox_out, class_out, obj_out, targets):
        # Placeholder for a more complex loss function based on regression and classification
        bbox_loss = nn.functional.mse_loss(bbox_out, targets["bbox"])
        class_loss = nn.CrossEntropyLoss()(class_out, targets["class"])
        obj_loss = nn.BCEWithLogitsLoss()(obj_out, targets["obj"])
        
        total_loss = bbox_loss + class_loss + obj_loss
        return total_loss
