# Hugging Face Deployment Guide

## Quick Deployment Steps

### 1. Login to Hugging Face
```bash
hf auth login
```
Enter your Hugging Face token when prompted.

### 2. Create the Space
```bash
hf repos create sidrayounas123/skin-disease-backend --type space --space-sdk docker
```

### 3. Push to Hugging Face
```bash
git remote add hf https://huggingface.co/spaces/sidrayounas123/skin-disease-backend
git push hf main
```

### 4. Monitor Deployment
Visit: https://huggingface.co/spaces/sidrayounas123/skin-disease-backend

## What's Included in This Deployment

### Features
- **Skin Detection**: Prevents misclassification of non-skin images
- **Dual Model Support**: Both model1.pth and model2.pth with proper error handling
- **Confidence-based Filtering**: 50% confidence threshold for disease predictions
- **Firebase Integration**: Graceful fallback if not available
- **Robust Error Handling**: All endpoints handle missing dependencies

### API Endpoints
- `POST /predict/dataset1` - Skin disease prediction with Model 1
- `POST /predict/dataset2` - Skin disease prediction with Model 2
- `GET /status` - Check model loading status
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /scans/{user_id}` - User scan history

### Response Format
For valid skin images:
```json
{
    "success": true,
    "predicted_disease": "Disease Name",
    "confidence_percent": 85.5,
    "is_valid_skin_image": true,
    "severity": "Moderate",
    "description": "Disease description",
    "precautions": ["Precaution 1", "Precaution 2"],
    "initial_treatment": ["Treatment 1", "Treatment 2"]
}
```

For invalid/non-skin images:
```json
{
    "success": false,
    "predicted_disease": "Unknown",
    "message": "Image does not appear to be a skin condition. Please upload a clear skin image.",
    "confidence_percent": 25.3,
    "is_valid_skin_image": false
}
```

## Fixes Applied

### 1. Skin Detector Issues Fixed
- Fixed `load_skin_detector()` to load weights from `weights/model2.pth`
- Added proper error handling and fallback to heuristics
- Integrated skin detection into both prediction endpoints

### 2. Model Loading Issues Fixed
- Model 1: Properly loads from `weights/model1.pth` (6 classes)
- Model 2: Updated to handle 15 classes from checkpoint
- Added graceful error handling for missing weights

### 3. Misclassification Prevention
- Added skin detection check before disease prediction
- 0.7 threshold for skin detection
- Confidence-based filtering (50% threshold)
- Proper response format with `is_valid_skin_image` field

## Deployment Time
- **Expected build time**: 5-10 minutes
- **Hardware**: CPU Basic (default) or upgrade as needed
- **Storage**: ~700MB for model weights

## Testing the Deployment
Once deployed, test with:
1. Valid skin images (should return disease predictions)
2. Non-skin images (should be rejected with proper message)
3. Low confidence images (should be rejected)

## Troubleshooting
If deployment fails:
1. Check the build logs on Hugging Face
2. Verify all files are pushed correctly
3. Check Dockerfile and requirements.txt
4. Ensure model weights are included

## Support
For issues:
- Check Hugging Face Space logs
- Verify model weights are in `weights/` directory
- Ensure all dependencies are in requirements.txt
