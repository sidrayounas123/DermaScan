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
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Log original image properties for bias detection
        print(f"Original image: mode={image.mode}, size={image.size}")
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Check for potential bias-inducing characteristics
        width, height = image.size
        aspect_ratio = width / height
        
        # Log preprocessing steps for bias analysis
        print(f"Image properties: size={width}x{height}, aspect_ratio={aspect_ratio:.2f}")
        
        # Apply preprocessing transform
        tensor = preprocess(image)
        
        # Log tensor statistics for bias detection
        print(f"Tensor stats: mean={tensor.mean():.4f}, std={tensor.std():.4f}, min={tensor.min():.4f}, max={tensor.max():.4f}")
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        print("Preprocessing completed successfully")
        return tensor
        
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Error preprocessing image: {e}")
