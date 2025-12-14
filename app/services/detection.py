"""
Object Detection Service
"""
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_Weights
from PIL import Image
import io
from typing import Dict, List
from loguru import logger
import numpy as np


class DetectionService:
    """Service for object detection using pretrained models"""

    def __init__(self, model_name: str = "fasterrcnn_resnet50"):
        """
        Initialize detection service with a pretrained model
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.classes = self._load_coco_classes()
        logger.info(f"Loaded {model_name} on {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """Load pretrained detection model"""
        if self.model_name == "fasterrcnn_resnet50":
            model = models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        elif self.model_name == "retinanet_resnet50":
            model = models.detection.retinanet_resnet50_fpn(
                weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
            )
        else:
            model = models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def _load_coco_classes(self) -> List[str]:
        """Load COCO class labels"""
        # Simplified COCO classes
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5,
        max_detections: int = 100
    ) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image_bytes: Image file bytes
            confidence_threshold: Minimum confidence score for detections
            max_detections: Maximum number of detections to return
            
        Returns:
            List of detections with bounding boxes, labels, and scores
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transform(image).to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model([input_tensor])[0]
            
            # Filter by confidence threshold
            scores = predictions['scores'].cpu().numpy()
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            mask = scores >= confidence_threshold
            scores = scores[mask]
            boxes = boxes[mask]
            labels = labels[mask]
            
            # Limit to max detections
            if len(scores) > max_detections:
                scores = scores[:max_detections]
                boxes = boxes[:max_detections]
                labels = labels[:max_detections]
            
            detections = []
            for score, box, label in zip(scores, boxes, labels):
                detections.append({
                    "label": self.classes[label] if label < len(self.classes) else f"class_{label}",
                    "confidence": float(score),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            raise


# Global instances cache
_detection_services = {}


def get_detection_service(model_name: str = "fasterrcnn_resnet50") -> DetectionService:
    """Get or create detection service instance"""
    global _detection_services
    if model_name not in _detection_services:
        _detection_services[model_name] = DetectionService(model_name)
    return _detection_services[model_name]
