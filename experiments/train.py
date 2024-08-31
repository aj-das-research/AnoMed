import torch
import torch.optim as optim
import torch.nn as nn
from models.encoder import ResNet50Encoder
from models.sib import ScaleInvariantBottleneck
from models.fa import FeatureAggregator
from models.plo import PseudoLabelOptimizer
from models.detector import DetectionHead
from data.loaders import get_train_dataloader
from utils.logger import get_logger
from utils.metrics import calculate_ap50, calculate_ap50_95

# Training loop
def train_model(dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train_logger", "train.log")
    
    encoder = ResNet50Encoder().to(device)
    sib = ScaleInvariantBottleneck(in_channels=2048).to(device)
    fa = FeatureAggregator(in_channels_list=[512, 1024, 2048]).to(device)
    plo = PseudoLabelOptimizer().to(device)
    detector = DetectionHead(in_channels=2048, num_classes=config["num_classes"]).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + 
                           list(sib.parameters()) + 
                           list(fa.parameters()) + 
                           list(detector.parameters()), 
                           lr=config["learning_rate"])
    
    for epoch in range(config["epochs"]):
        encoder.train()
        sib.train()
        fa.train()
        detector.train()

        running_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), {k: v.to(device) for k, v in targets.items()}
            
            features = encoder(images)
            features = sib(features)
            features = fa(features)
            bbox_out, class_out, obj_out = detector(features)
            
            confident, hesitant, scrap = plo(targets["label"])
            loss = detector.compute_loss(bbox_out, class_out, obj_out, targets)
            unsupervised_loss = plo.calculate_loss(bbox_out, targets["bbox"], confident, hesitant)
            
            total_loss = loss + unsupervised_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        logger.info(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss / len(dataloader)}")
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss / len(dataloader)}")

if __name__ == "__main__":
    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "num_classes": 5,
        "batch_size": 16
    }
    
    train_dataloader = get_train_dataloader(
        root_dir="path/to/images", 
        annotations_file="path/to/annotations.json", 
        batch_size=config["batch_size"]
    )
    
    train_model(train_dataloader, config)
