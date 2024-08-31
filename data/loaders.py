import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        bbox = torch.tensor(self.annotations[idx]['bbox'], dtype=torch.float32)
        label = torch.tensor(self.annotations[idx]['label'], dtype=torch.int64)
        target = {
            'bbox': bbox,
            'label': label
        }

        if self.transform:
            image = self.transform(image)

        return image, target

def get_train_dataloader(root_dir, annotations_file, batch_size=16, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir=root_dir, annotations_file=annotations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_val_dataloader(root_dir, annotations_file, batch_size=16, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir=root_dir, annotations_file=annotations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_test_dataloader(root_dir, annotations_file, batch_size=16, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir=root_dir, annotations_file=annotations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
