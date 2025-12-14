"""
Image Classification Service
"""
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from typing import Dict, List, Tuple
from loguru import logger


class ClassificationService:
    """Service for image classification using pretrained models"""

    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize classification service with a pretrained model
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.classes = self._load_imagenet_classes()
        logger.info(f"Loaded {model_name} on {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """Load pretrained model"""
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v3":
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_imagenet_classes(self) -> List[str]:
        """Load ImageNet class labels"""
        # Simplified list - in production, load from a file
        return [f"class_{i}" for i in range(1000)]

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Predict image class
        
        Args:
            image_bytes: Image file bytes
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class labels and confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = [
                {
                    "class": self.classes[idx.item()],
                    "confidence": prob.item(),
                    "class_id": idx.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise

    def transfer_learning(self, num_classes: int) -> torch.nn.Module:
        """
        Prepare model for transfer learning
        
        Args:
            num_classes: Number of output classes for the new task
            
        Returns:
            Model with modified output layer
        """
        model = self._load_model()
        
        # Modify the final layer for transfer learning
        if "resnet" in self.model_name:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)
        elif "mobilenet" in self.model_name:
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
        elif "efficientnet" in self.model_name:
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
        
        return model


# Global instance
_classification_service = None


def get_classification_service(model_name: str = "resnet50") -> ClassificationService:
    """Get or create classification service instance"""
    global _classification_service
    if _classification_service is None:
        _classification_service = ClassificationService(model_name)
    return _classification_service
