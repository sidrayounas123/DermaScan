import torch
import torch.nn as nn
import os
import math

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm library not available, will try custom ViT")

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names for Dataset 1 (common skin diseases)
CLASS_NAMES_1 = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

# Global model instance
_model1 = None

def load_model1():
    """Load Model 1 (Dataset 1) - Vision Transformer"""
    global _model1
    if _model1 is not None:
        return _model1
    
    try:
        current_dir = os.getcwd()
        weights_path = os.path.join(current_dir, "weights", "model1.pth")
        
        if not os.path.exists(weights_path):
            print(f"Model 1 weights not found at {weights_path}")
            return None
        
        # Load checkpoint first to inspect architecture
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        
        # Check if it's a timm model with 'model' key or direct state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Inspect keys to determine architecture
        first_key = list(state_dict.keys())[0] if state_dict else ""
        
        if 'backbone' in first_key or 'patch_embed' in first_key:
            # This is a Vision Transformer
            print("Detected Vision Transformer architecture")
            
            if TIMM_AVAILABLE:
                # Try to load using timm
                model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES_1))
            else:
                # Create a simple ViT-like model structure
                model = nn.Module()
                model.backbone = nn.Module()
                model.backbone.patch_embed = nn.Module()
                model.backbone.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
                model.backbone.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
                model.backbone.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
                model.backbone.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=768, nhead=12) for _ in range(12)])
                model.backbone.norm = nn.LayerNorm(768)
                model.head = nn.Linear(768, len(CLASS_NAMES_1))
        else:
            # Fallback to ResNet
            print("Trying ResNet architecture as fallback")
            import torchvision.models as models
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES_1))
        
        # Load weights with strict=False to allow partial loading
        model.load_state_dict(state_dict, strict=False)
        
        model = model.to(DEVICE)
        model.eval()
        _model1 = model
        print("Model 1 loaded successfully")
        return _model1
        
    except Exception as e:
        print(f"Error loading Model 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict1(image_tensor):
    """Make prediction using Model 1"""
    model = load_model1()
    if model is None:
        raise Exception("Model 1 not available")
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            disease = CLASS_NAMES_1[predicted.item()]
            confidence_value = confidence.item()
            all_probs = probabilities[0].cpu().numpy().tolist()
            
            return disease, confidence_value, all_probs
            
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def get_gradcam(image_tensor):
    """Generate GradCAM heatmap for Model 1"""
    model = load_model1()
    if model is None:
        return None
    
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)
    
    # Register hook based on model architecture
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        # Vision Transformer - hook on last transformer block
        hook = model.backbone.blocks[-1].register_forward_hook(forward_hook)
    elif hasattr(model, 'layer4'):
        # ResNet - hook on last convolutional layer
        hook = model.layer4[-1].register_forward_hook(forward_hook)
    else:
        print("Unsupported architecture for GradCAM")
        return None
    
    model.eval()
    output = model(image_tensor.to(DEVICE))
    pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()
    
    hook.remove()
    
    if not gradients or not activations:
        return None
    
    grad = gradients[0]
    act = activations[0]
    
    # Handle different tensor shapes
    if grad.dim() == 3:  # ViT: [batch, seq_len, hidden_dim]
        # For ViT, use attention weights
        weights = grad.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        cam = (weights * act).sum(dim=-1)  # [batch, seq_len]
        cam = cam[:, 1:]  # Remove CLS token
        # Reshape to square (assuming 14x14 patches)
        size = int(math.sqrt(cam.shape[1]))
        cam = cam.reshape(1, 1, size, size)
    elif grad.dim() == 4:  # CNN: [batch, channels, height, width]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
    else:
        print("Unsupported tensor shape for GradCAM")
        return None
    
    # Resize to input size
    import torch.nn.functional as F
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Convert to numpy and normalize
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam
