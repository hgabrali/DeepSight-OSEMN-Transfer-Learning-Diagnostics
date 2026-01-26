import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def run_diagnostic_plots(model, dataloader, device, classes, filename="confusion_matrix"):
    """
    Saves a confusion matrix to identify class-wise errors.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"Confusion Matrix: {filename}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'notebooks/{filename}.png')
    plt.close()
    print(f"📊 {filename}.png saved to notebooks/ folder.")

def run_grad_cam_visualization(model, dataset, device, classes, num_images=5):
    """
    Generates heatmaps showing where the model is looking.
    """
    model.eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    plt.figure(figsize=(15, num_images * 3))
    for i in range(num_images):
        idx = np.random.randint(0, len(dataset))
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        outputs = model(input_tensor)
        pred_label = outputs.argmax(dim=1).item()
        
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        img_show = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
        
        visualization = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)
        
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(img_show)
        plt.title(f"Original: {classes[label]}")
        plt.axis('off')
        
        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM (Pred: {classes[pred_label]})")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/grad_cam_results.png')
    plt.close()
    print("🔥 Grad-CAM heatmaps saved to notebooks/grad_cam_results.png")