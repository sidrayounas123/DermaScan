import io
from PIL import Image
import torch
import torchvision.transforms as transforms

# Preprocess transform for model input (matching DeiT training pipeline)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger for center crop
    transforms.CenterCrop(224),     # Standard center crop for DeiT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization for DeiT
])

def preprocess_image(file_bytes: bytes):
    """
    Preprocess uploaded image bytes for model inference
    
    Args:
        file_bytes: Raw image bytes from uploaded file
        
    Returns:
        Preprocessed tensor ready for model input
    """
    try:
        # Try multiple methods to open image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB (handles PNG with alpha, grayscale, etc)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        tensor = preprocess(image).unsqueeze(0)
        return tensor
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")
