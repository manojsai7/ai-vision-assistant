"""
Object Detection API Routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from loguru import logger

from app.services.detection import get_detection_service

router = APIRouter()


@router.post("/predict")
async def detect_objects(
    file: UploadFile = File(...),
    model: Optional[str] = Query("fasterrcnn_resnet50", description="Model to use: fasterrcnn_resnet50, retinanet_resnet50"),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    max_detections: int = Query(100, ge=1, le=500, description="Maximum number of detections")
):
    """
    Detect objects in an image using a pretrained model
    
    Args:
        file: Image file to process
        model: Model name to use for detection
        confidence_threshold: Minimum confidence score for detections
        max_detections: Maximum number of detections to return
        
    Returns:
        List of detected objects with bounding boxes and confidence scores
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get detection service
        service = get_detection_service(model)
        
        # Perform detection
        detections = service.detect(image_bytes, confidence_threshold, max_detections)
        
        return {
            "model": model,
            "detections": detections,
            "num_detections": len(detections),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Detection endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available object detection models"""
    return {
        "models": [
            {
                "name": "fasterrcnn_resnet50",
                "description": "Faster R-CNN with ResNet-50 backbone, pretrained on COCO",
                "type": "Two-stage detector"
            },
            {
                "name": "retinanet_resnet50",
                "description": "RetinaNet with ResNet-50 backbone, pretrained on COCO",
                "type": "One-stage detector"
            }
        ]
    }
