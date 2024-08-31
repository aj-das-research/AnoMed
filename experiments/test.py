import torch
from models.encoder import ResNet50Encoder
from models.sib import ScaleInvariantBottleneck
from models.fa import FeatureAggregator
from models.detector import DetectionHead
from data.loaders import get_test_dataloader
from utils.metrics import calculate_ap50, calculate_ap50_95

def test_model(dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = ResNet50Encoder().to(device)
    sib = ScaleInvariantBottleneck(in_channels=2048).to(device)
    fa = FeatureAggregator(in_channels_list=[512, 1024, 2048]).to(device)
    detector = DetectionHead(in_channels=2048, num_classes=config["num_classes"]).to(device)
    
    encoder.eval()
    sib.eval()
    fa.eval()
    detector.eval()
    
    total_ap50 = 0.0
    total_ap50_95 = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            features = encoder(images)
            features = sib(features)
            features = fa(features)
            bbox_out, class_out, obj_out = detector(features)
            
            ap50 = calculate_ap50(bbox_out, targets["bbox"])
            ap50_95 = calculate_ap50_95(bbox_out, targets["bbox"])
            
            total_ap50 += ap50
            total_ap50_95 += ap50_95

        avg_ap50 = total_ap50 / num_batches
        avg_ap50_95 = total_ap50_95 / num_batches
        
        print(f"Test AP50: {avg_ap50}, AP50:95: {avg_ap50_95}")

if __name__ == "__main__":
    config = {
        "num_classes": 5,
        "batch_size": 16
    }
    
    test_dataloader = get_test_dataloader(
        root_dir="path/to/images", 
        annotations_file="path/to/annotations.json", 
        batch_size=config["batch_size"]
    )
    
    test_model(test_dataloader, config)
