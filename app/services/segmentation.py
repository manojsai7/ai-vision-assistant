"""
Image Segmentation Service
"""
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, FCN_ResNet50_Weights
from PIL import Image
import io
from typing import Dict, List
from loguru import logger
import numpy as np


class SegmentationService:
    """Service for image segmentation using pretrained models"""

    def __init__(self, model_name: str = "deeplabv3_resnet50"):
        """
        Initialize segmentation service with a pretrained model
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.classes = self._load_segmentation_classes()
        logger.info(f"Loaded {model_name} on {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """Load pretrained segmentation model"""
        if self.model_name == "deeplabv3_resnet50":
            model = models.segmentation.deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            )
        elif self.model_name == "fcn_resnet50":
            model = models.segmentation.fcn_resnet50(
                weights=FCN_ResNet50_Weights.DEFAULT
            )
        else:
            model = models.segmentation.deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            )
        
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_segmentation_classes(self) -> List[str]:
        """Load segmentation class labels (Pascal VOC / COCO)"""
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def segment(self, image_bytes: bytes) -> Dict:
        """
        Perform semantic segmentation on an image
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            Dictionary with segmentation mask and class statistics
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = image.size
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                output_predictions = output.argmax(0).cpu().numpy()
            
            # Resize to original size
            mask_image = Image.fromarray(output_predictions.astype(np.uint8))
            mask_image = mask_image.resize(original_size, Image.NEAREST)
            mask = np.array(mask_image)
            
            # Calculate class statistics
            unique_classes, counts = np.unique(mask, return_counts=True)
            total_pixels = mask.shape[0] * mask.shape[1]
            
            class_stats = []
            for class_id, count in zip(unique_classes, counts):
                if class_id < len(self.classes):
                    class_name = self.classes[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                class_stats.append({
                    "class": class_name,
                    "class_id": int(class_id),
                    "pixel_count": int(count),
                    "percentage": float(count / total_pixels * 100)
                })
            
            return {
                "width": mask.shape[1],
                "height": mask.shape[0],
                "classes": class_stats,
                "mask_shape": mask.shape,
                "num_classes": len(unique_classes)
            }
            
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            raise


# Global instances cache
_segmentation_services = {}


def get_segmentation_service(model_name: str = "deeplabv3_resnet50") -> SegmentationService:
    """Get or create segmentation service instance"""
    global _segmentation_services
    if model_name not in _segmentation_services:
        _segmentation_services[model_name] = SegmentationService(model_name)
    return _segmentation_services[model_name]
