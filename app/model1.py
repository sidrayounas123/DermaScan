import torch
import torch.nn as nn
import torchvision.models as models
import os

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
    """Load Model 1 (Dataset 1)"""
    global _model1
    if _model1 is not None:
        return _model1
    
    try:
        current_dir = os.getcwd()
        weights_path = os.path.join(current_dir, "weights", "model1.pth")
        
        if not os.path.exists(weights_path):
            print(f"Model 1 weights not found at {weights_path}")
            return None
        
        # Load a ResNet50 model (common for skin disease classification)
        model = models.resnet50(pretrained=False)
        
        # Modify the final layer for our number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES_1))
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        _model1 = model
        print("Model 1 loaded successfully")
        return _model1
        
    except Exception as e:
        print(f"Error loading Model 1: {str(e)}")
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
    
    # Register hook on the last convolutional layer (layer4)
    hook = model.layer4[-1].register_forward_hook(forward_hook)
    
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
    
    # Global average pooling of gradients
    weights = grad.mean(dim=(2, 3), keepdim=True)
    
    # Weighted combination of activation maps
    cam = (weights * act).sum(dim=1, keepdim=True)
    
    # Apply ReLU
    cam = torch.clamp(cam, min=0)
    
    # Resize to input size
    import torch.nn.functional as F
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Convert to numpy and normalize
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam
