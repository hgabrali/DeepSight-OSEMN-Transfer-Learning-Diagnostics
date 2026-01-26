import torchvision.transforms as T

def get_scrub_pipeline(method="bicubic"):
    """
    Scrub & Engineering: Resizing and Normalization.
    """
    interpolation = T.InterpolationMode.BICUBIC if method == "bicubic" else T.InterpolationMode.BILINEAR
    
    # Caffe-style normalization (approximate values for ImageNet)
    pipeline = T.Compose([
        T.Resize((224, 224), interpolation=interpolation),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Scrub Pipeline Ready: Using {method.upper()} interpolation.")
    return pipeline