"""
Tests for evaluation metrics
"""
import pytest
import numpy as np
from app.utils.metrics import calculate_iou, calculate_f1, calculate_segmentation_metrics


def test_calculate_iou_perfect_overlap():
    """Test IoU with perfect overlap"""
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    iou = calculate_iou(box1, box2)
    assert iou == 1.0


def test_calculate_iou_no_overlap():
    """Test IoU with no overlap"""
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    iou = calculate_iou(box1, box2)
    assert iou == 0.0


def test_calculate_iou_partial_overlap():
    """Test IoU with partial overlap"""
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = calculate_iou(box1, box2)
    assert 0 < iou < 1


def test_calculate_f1():
    """Test F1 score calculation"""
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    
    result = calculate_f1(y_true, y_pred)
    
    assert "f1_score" in result
    assert "precision" in result
    assert "recall" in result
    assert "accuracy" in result
    assert 0 <= result["f1_score"] <= 1
    assert 0 <= result["accuracy"] <= 1


def test_calculate_segmentation_metrics():
    """Test segmentation metrics calculation"""
    pred_mask = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    gt_mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    
    result = calculate_segmentation_metrics(pred_mask, gt_mask)
    
    assert "iou" in result
    assert "dice_coefficient" in result
    assert "pixel_accuracy" in result
    assert 0 <= result["iou"] <= 1
    assert 0 <= result["dice_coefficient"] <= 1
    assert 0 <= result["pixel_accuracy"] <= 1
