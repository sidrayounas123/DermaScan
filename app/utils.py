import io
from PIL import Image
import torch
import torchvision.transforms as transforms

# Preprocess transform for model input
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transform
        tensor = preprocess(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")
