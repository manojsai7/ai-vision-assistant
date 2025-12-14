"""
Computer Vision Evaluation Metrics
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truth: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    # Simplified mAP calculation
    # In production, use a more robust implementation
    
    aps = []
    
    for class_id in range(num_classes):
        # Get predictions and ground truth for this class
        class_preds = [
            p for p in predictions
            if p.get('label_id') == class_id or p.get('labels') == class_id
        ]
        class_gt = [
            g for g in ground_truth
            if g.get('label_id') == class_id or g.get('labels') == class_id
        ]
        
        if len(class_gt) == 0:
            continue
        
        # Sort predictions by confidence
        class_preds = sorted(
            class_preds,
            key=lambda x: x.get('confidence', x.get('scores', 0)),
            reverse=True
        )
        
        # Calculate precision and recall
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred in class_preds:
            pred_box = pred.get('bbox', pred.get('boxes'))
            if isinstance(pred_box, dict):
                pred_box = [pred_box['x1'], pred_box['y1'], pred_box['x2'], pred_box['y2']]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gt):
                if gt_idx in matched_gt:
                    continue
                
                gt_box = gt.get('bbox', gt.get('boxes'))
                if isinstance(gt_box, dict):
                    gt_box = [gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']]
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(class_gt) if len(class_gt) > 0 else 0
        
        # Simplified AP calculation (Note: This is a basic approximation.
        # For production use, implement proper precision-recall curve integration
        # or use sklearn.metrics.average_precision_score)
        ap = precision * recall
        aps.append(ap)
    
    return {
        "mAP": np.mean(aps) if aps else 0.0,
        "num_classes_evaluated": len(aps)
    }


def calculate_f1(y_true: List[int], y_pred: List[int], average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate F1 score, precision, and recall
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted')
        
    Returns:
        Dictionary with F1, precision, recall, and accuracy
    """
    return {
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }


def calculate_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate segmentation metrics (IoU, dice coefficient)
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        
    Returns:
        Dictionary with IoU and dice coefficient
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    iou = intersection / union if union > 0 else 0.0
    
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    return {
        "iou": float(iou),
        "dice_coefficient": float(dice),
        "pixel_accuracy": float(np.mean(pred_mask == gt_mask))
    }
