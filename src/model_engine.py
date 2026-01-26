import torch
import torch.nn as nn
from torchvision import models

def build_resnet18_model(num_classes=10):
    """
    Model: ResNet18 Initialization & Head Reconstruction.
    """
    # 1. Load Pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 2. Phase 1: Frozen Base (Dondurma)
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. Head Reconstruction (Yeni Sınıflandırıcı Ekleme)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    
    print("✅ Model: ResNet18 loaded. Base layers frozen, Head reconstructed.")
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training Phase Helper.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def unfreeze_layers(model):
    """
    Fine-Tuning: Unfreezing Stage 3 & 4 of ResNet18.
    """
    for name, child in model.named_children():
        if name in ['layer3', 'layer4']:
            print(f"🔓 Unfreezing {name} for Fine-Tuning...")
            for param in child.parameters():
                param.requires_grad = True
    return model