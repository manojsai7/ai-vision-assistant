"""
Image Segmentation API Routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from loguru import logger

from app.services.segmentation import get_segmentation_service

router = APIRouter()


@router.post("/predict")
async def segment_image(
    file: UploadFile = File(...),
    model: Optional[str] = Query("deeplabv3_resnet50", description="Model to use: deeplabv3_resnet50, fcn_resnet50")
):
    """
    Perform semantic segmentation on an image
    
    Args:
        file: Image file to segment
        model: Model name to use for segmentation
        
    Returns:
        Segmentation results with class statistics
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get segmentation service
        service = get_segmentation_service(model)
        
        # Perform segmentation
        result = service.segment(image_bytes)
        
        return {
            "model": model,
            "filename": file.filename,
            **result
        }
        
    except Exception as e:
        logger.error(f"Segmentation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available segmentation models"""
    return {
        "models": [
            {
                "name": "deeplabv3_resnet50",
                "description": "DeepLabV3 with ResNet-50 backbone, pretrained on COCO",
                "output": "Semantic segmentation"
            },
            {
                "name": "fcn_resnet50",
                "description": "Fully Convolutional Network with ResNet-50 backbone",
                "output": "Semantic segmentation"
            }
        ]
    }
