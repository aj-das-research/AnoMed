import torch

def calculate_ap50(pred_boxes, true_boxes):
    # Placeholder for AP50 calculation
    iou_threshold = 0.5
    ious = compute_iou(pred_boxes, true_boxes)
    return (ious >= iou_threshold).float().mean()

def calculate_ap50_95(pred_boxes, true_boxes):
    # Placeholder for AP50:95 calculation
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    ap50_95 = 0.0
    for threshold in iou_thresholds:
        ious = compute_iou(pred_boxes, true_boxes)
        ap50_95 += (ious >= threshold).float().mean()
    return ap50_95 / len(iou_thresholds)

def compute_iou(pred_boxes, true_boxes):
    # Basic IoU calculation between predicted and true boxes
    inter_x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    union_area = pred_area + true_area - inter_area
    
    return inter_area / torch.clamp(union_area, min=1e-6)
