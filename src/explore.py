import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_exploratory_analysis(dataset):
    """
    Explore: Class Balance ve Görüntü Önizleme.
    """
    # Etiketleri alalım
    labels = [label for _, label in dataset]
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 1. Class Balance (Sınıf Dengesi)
    plt.figure(figsize=(10, 5))
    sns.countplot(x=labels)
    plt.xticks(ticks=range(10), labels=classes, rotation=45)
    plt.title("Class Distribution (Stratified Subset)")
    plt.savefig('notebooks/class_distribution.png') # Grafiği kaydet
    print("📈 EDA: Class distribution plot saved to notebooks/ folder.")

    # 2. Örnek Görüntüleri Göster (Opsiyonel - Doğrulama için)
    print("✅ EDA: Analysis complete.")