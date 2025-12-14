"""
Evaluation API Routes
"""
from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict
from loguru import logger

from app.utils.metrics import calculate_iou, calculate_map, calculate_f1, calculate_segmentation_metrics
import numpy as np

router = APIRouter()


@router.post("/iou")
async def compute_iou(
    box1: List[float] = Body(..., description="First bounding box [x1, y1, x2, y2]"),
    box2: List[float] = Body(..., description="Second bounding box [x1, y1, x2, y2]")
):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First bounding box coordinates
        box2: Second bounding box coordinates
        
    Returns:
        IoU score
    """
    try:
        iou = calculate_iou(box1, box2)
        return {"iou": iou}
    except Exception as e:
        logger.error(f"IoU calculation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/map")
async def compute_map(
    predictions: List[Dict] = Body(..., description="List of predictions"),
    ground_truth: List[Dict] = Body(..., description="List of ground truth annotations"),
    iou_threshold: float = Body(0.5, description="IoU threshold for matching"),
    num_classes: int = Body(80, description="Number of classes")
):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching predictions to ground truth
        num_classes: Number of object classes
        
    Returns:
        mAP score and additional metrics
    """
    try:
        result = calculate_map(predictions, ground_truth, iou_threshold, num_classes)
        return result
    except Exception as e:
        logger.error(f"mAP calculation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/f1")
async def compute_f1(
    y_true: List[int] = Body(..., description="True labels"),
    y_pred: List[int] = Body(..., description="Predicted labels"),
    average: str = Body("weighted", description="Averaging method: micro, macro, weighted")
):
    """
    Calculate F1 score, precision, recall, and accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        F1 score, precision, recall, and accuracy
    """
    try:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        result = calculate_f1(y_true, y_pred, average)
        return result
    except Exception as e:
        logger.error(f"F1 calculation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/segmentation-metrics")
async def compute_segmentation_metrics(
    pred_mask: List[List[int]] = Body(..., description="Predicted segmentation mask"),
    gt_mask: List[List[int]] = Body(..., description="Ground truth segmentation mask")
):
    """
    Calculate segmentation metrics (IoU, Dice coefficient, pixel accuracy)
    
    Args:
        pred_mask: Predicted segmentation mask as 2D array
        gt_mask: Ground truth segmentation mask as 2D array
        
    Returns:
        Segmentation metrics
    """
    try:
        pred_array = np.array(pred_mask)
        gt_array = np.array(gt_mask)
        
        if pred_array.shape != gt_array.shape:
            raise ValueError("Predicted and ground truth masks must have the same shape")
        
        result = calculate_segmentation_metrics(pred_array, gt_array)
        return result
    except Exception as e:
        logger.error(f"Segmentation metrics calculation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics")
async def list_metrics():
    """List available evaluation metrics"""
    return {
        "metrics": [
            {
                "name": "IoU",
                "description": "Intersection over Union for bounding boxes",
                "endpoint": "/api/v1/evaluate/iou"
            },
            {
                "name": "mAP",
                "description": "mean Average Precision for object detection",
                "endpoint": "/api/v1/evaluate/map"
            },
            {
                "name": "F1",
                "description": "F1 score, precision, recall for classification",
                "endpoint": "/api/v1/evaluate/f1"
            },
            {
                "name": "Segmentation Metrics",
                "description": "IoU, Dice coefficient, pixel accuracy for segmentation",
                "endpoint": "/api/v1/evaluate/segmentation-metrics"
            }
        ]
    }
