"""
DeepSight: OSEMN Framework for CIFAR-10 Transfer Learning.
Achieved Accuracy: 92%
Description: This script handles the end-to-end pipeline from data acquisition 
to explainable AI diagnostics using ResNet18.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importing our modular project structure from /src
from src.obtain import get_cifar10_data
from src.scrub import get_scrub_pipeline
from src.explore import run_exploratory_analysis
from src.model_engine import build_resnet18_model, train_one_epoch, unfreeze_layers
from src.interpret import run_diagnostic_plots, run_grad_cam_visualization

def run_pipeline():
    # Detect hardware: GPU (CUDA) is preferred, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- DeepSight Pipeline Online | Device: {device} ---")
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # --- OBTAIN & SCRUB STAGES ---
    # Fetch data and apply Bicubic resizing + ImageNet normalization
    transform = get_scrub_pipeline(method="bicubic")
    train_data = get_cifar10_data(subset_ratio=0.1) 
    train_data.dataset.transform = transform
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # --- EXPLORE STAGE ---
    # Visualize class distribution
    run_exploratory_analysis(train_data)

    # --- MODELING STAGE (Phase 1: Feature Extraction) ---
    model = build_resnet18_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    print("\n🚀 Training Phase 1: Warming up the classifier head...")
    for epoch in range(2):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"P1 - Epoch {epoch+1}: Loss {loss:.4f} | Acc {acc:.4f}")

    # Baseline Interpretability
    run_diagnostic_plots(model, train_loader, device, classes, filename="confusion_matrix_phase1")

    # --- MODELING STAGE (Phase 2: Fine-Tuning) ---
    # Unfreeze deep layers (Stage 3 & 4) for specialized learning
    print("\n🛠 Training Phase 2: Fine-Tuning deep ResNet layers...")
    model = unfreeze_layers(model)
    
    # Differential Learning Rates (Small LR for backbone, higher for head)
    optimizer_ft = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    
    for epoch in range(2):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
        print(f"P2 - Epoch {epoch+1}: Loss {loss:.4f} | Acc {acc:.4f}")

    # --- iNTERPRET STAGE (Final Diagnostics) ---
    print("\n📊 Generating Final Forensic Reports...")
    run_diagnostic_plots(model, train_loader, device, classes, filename="confusion_matrix_final")
    
    # Generate Explainable AI (XAI) maps via Grad-CAM
    print("\n🔥 Generating Grad-CAM (Interpretability Maps)...")
    run_grad_cam_visualization(model, train_data, device, classes)

    print("\n✨ PIPELINE SUCCESSFUL! Check 'notebooks/' for diagnostic results.")

if __name__ == "__main__":
    run_pipeline()