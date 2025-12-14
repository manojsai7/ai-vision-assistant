"""
Classification API Routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from loguru import logger

from app.services.classification import get_classification_service

router = APIRouter()


@router.post("/predict")
async def classify_image(
    file: UploadFile = File(...),
    model: Optional[str] = Query("resnet50", description="Model to use: resnet50, resnet18, mobilenet_v3, efficientnet_b0"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top predictions to return")
):
    """
    Classify an image using a pretrained model
    
    Args:
        file: Image file to classify
        model: Model name to use for classification
        top_k: Number of top predictions to return
        
    Returns:
        List of predictions with class labels and confidence scores
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get classification service
        service = get_classification_service(model)
        
        # Perform classification
        predictions = service.predict(image_bytes, top_k)
        
        return {
            "model": model,
            "predictions": predictions,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Classification endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available classification models"""
    return {
        "models": [
            {
                "name": "resnet50",
                "description": "ResNet-50 pretrained on ImageNet",
                "parameters": "25.6M"
            },
            {
                "name": "resnet18",
                "description": "ResNet-18 pretrained on ImageNet",
                "parameters": "11.7M"
            },
            {
                "name": "mobilenet_v3",
                "description": "MobileNetV3-Large pretrained on ImageNet",
                "parameters": "5.4M"
            },
            {
                "name": "efficientnet_b0",
                "description": "EfficientNet-B0 pretrained on ImageNet",
                "parameters": "5.3M"
            }
        ]
    }


@router.post("/transfer-learning")
async def setup_transfer_learning(
    num_classes: int = Query(..., ge=2, le=1000, description="Number of output classes"),
    model: str = Query("resnet50", description="Base model to use")
):
    """
    Setup a model for transfer learning with custom number of classes
    
    Args:
        num_classes: Number of output classes for the new task
        model: Base model name
        
    Returns:
        Information about the modified model
    """
    try:
        service = get_classification_service(model)
        modified_model = service.transfer_learning(num_classes)
        
        return {
            "status": "success",
            "base_model": model,
            "num_classes": num_classes,
            "message": f"Model prepared for transfer learning with {num_classes} output classes"
        }
        
    except Exception as e:
        logger.error(f"Transfer learning endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
