"""
AI Vision Assistant - A comprehensive image understanding toolkit.

This module provides easy-to-use interfaces for:
- Image Classification: Identify what's in an image
- Object Detection: Find and locate objects in images
- Image Segmentation: Separate different regions/objects in images
"""

from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForImageSegmentation,
)
from typing import List, Dict, Tuple, Union, Any
import numpy as np


class VisionAssistant:
    """
    A unified AI vision assistant that provides classification, detection, and segmentation.
    
    Example:
        >>> assistant = VisionAssistant()
        >>> results = assistant.classify("path/to/image.jpg")
        >>> print(results)
    """
    
    def __init__(
        self,
        classification_model: str = "google/vit-base-patch16-224",
        detection_model: str = "facebook/detr-resnet-50",
        segmentation_model: str = "facebook/detr-resnet-50-panoptic",
    ):
        """
        Initialize the Vision Assistant with pre-trained models.
        
        Args:
            classification_model: HuggingFace model for image classification
            detection_model: HuggingFace model for object detection
            segmentation_model: HuggingFace model for image segmentation
        """
        self.classification_model_name = classification_model
        self.detection_model_name = detection_model
        self.segmentation_model_name = segmentation_model
        
        # Models are loaded lazily on first use
        self._classification_model = None
        self._classification_processor = None
        self._detection_model = None
        self._detection_processor = None
        self._segmentation_model = None
        self._segmentation_processor = None
    
    def _load_classification_model(self):
        """Lazy load the classification model."""
        if self._classification_model is None:
            self._classification_processor = AutoImageProcessor.from_pretrained(
                self.classification_model_name
            )
            self._classification_model = AutoModelForImageClassification.from_pretrained(
                self.classification_model_name
            )
    
    def _load_detection_model(self):
        """Lazy load the object detection model."""
        if self._detection_model is None:
            self._detection_processor = AutoImageProcessor.from_pretrained(
                self.detection_model_name
            )
            self._detection_model = AutoModelForObjectDetection.from_pretrained(
                self.detection_model_name
            )
    
    def _load_segmentation_model(self):
        """Lazy load the segmentation model."""
        if self._segmentation_model is None:
            self._segmentation_processor = AutoImageProcessor.from_pretrained(
                self.segmentation_model_name
            )
            self._segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                self.segmentation_model_name
            )
    
    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Load an image from path or return the PIL Image object.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image file path doesn't exist
            ValueError: If image cannot be loaded or converted
        """
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Image file not found: {image}")
            except Exception as e:
                raise ValueError(f"Failed to load image from {image}: {str(e)}")
        return image
    
    def classify(
        self, image: Union[str, Image.Image], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify an image and return top predictions.
        
        Args:
            image: Path to image file or PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with 'label' and 'score' keys
            
        Example:
            >>> assistant = VisionAssistant()
            >>> results = assistant.classify("cat.jpg", top_k=3)
            >>> print(results[0])
            {'label': 'tabby cat', 'score': 0.95}
        """
        self._load_classification_model()
        img = self._load_image(image)
        
        inputs = self._classification_processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self._classification_model(**inputs)
            logits = outputs.logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label_id = idx.item()
            label = self._classification_model.config.id2label.get(
                label_id, f"UNKNOWN_LABEL_{label_id}"
            )
            results.append({
                "label": label,
                "score": prob.item()
            })
        
        return results
    
    def detect(
        self, image: Union[str, Image.Image], threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Path to image file or PIL Image object
            threshold: Confidence threshold for detections
            
        Returns:
            List of dictionaries with 'label', 'score', and 'box' keys
            
        Example:
            >>> assistant = VisionAssistant()
            >>> results = assistant.detect("scene.jpg")
            >>> for obj in results:
            ...     print(f"{obj['label']}: {obj['score']:.2f} at {obj['box']}")
        """
        self._load_detection_model()
        img = self._load_image(image)
        
        inputs = self._detection_processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self._detection_model(**inputs)
        
        # Convert outputs to COCO API format
        target_sizes = torch.tensor([img.size[::-1]])
        results = self._detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_id = label.item()
            label_text = self._detection_model.config.id2label.get(
                label_id, f"UNKNOWN_LABEL_{label_id}"
            )
            box_coords = [round(i, 2) for i in box.tolist()]
            detections.append({
                "label": label_text,
                "score": score.item(),
                "box": box_coords  # [x_min, y_min, x_max, y_max]
            })
        
        return detections
    
    def segment(
        self, image: Union[str, Image.Image], threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Perform panoptic segmentation on an image.
        
        Args:
            image: Path to image file or PIL Image object
            threshold: Confidence threshold for segments
            
        Returns:
            Dictionary with 'segments' list and 'segmentation_map' array
            
        Example:
            >>> assistant = VisionAssistant()
            >>> results = assistant.segment("scene.jpg")
            >>> print(f"Found {len(results['segments'])} segments")
        """
        self._load_segmentation_model()
        img = self._load_image(image)
        
        inputs = self._segmentation_processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self._segmentation_model(**inputs)
        
        # Process panoptic segmentation
        target_sizes = torch.tensor([img.size[::-1]])
        results = self._segmentation_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        segments = []
        for segment_info in results["segments_info"]:
            label_id = segment_info["label_id"]
            label = self._segmentation_model.config.id2label.get(
                label_id, f"UNKNOWN_LABEL_{label_id}"
            )
            segments.append({
                "id": segment_info["id"],
                "label": label,
                "score": segment_info["score"],
                "area": segment_info.get("area", 0)
            })
        
        return {
            "segments": segments,
            "segmentation_map": results["segmentation"].numpy()
        }
    
    def analyze(
        self, image: Union[str, Image.Image], tasks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run multiple vision tasks on a single image.
        
        Args:
            image: Path to image file or PIL Image object
            tasks: List of tasks to perform. Options: ['classify', 'detect', 'segment']
                   If None, performs all tasks.
            
        Returns:
            Dictionary with results for each task
            
        Example:
            >>> assistant = VisionAssistant()
            >>> results = assistant.analyze("image.jpg", tasks=['classify', 'detect'])
            >>> print(results['classification'])
            >>> print(results['detection'])
        """
        if tasks is None:
            tasks = ['classify', 'detect', 'segment']
        
        results = {}
        
        if 'classify' in tasks:
            results['classification'] = self.classify(image)
        
        if 'detect' in tasks:
            results['detection'] = self.detect(image)
        
        if 'segment' in tasks:
            results['segmentation'] = self.segment(image)
        
        return results


# Convenience function for quick usage
def create_assistant(**kwargs) -> VisionAssistant:
    """
    Create a VisionAssistant instance with optional custom models.
    
    Args:
        **kwargs: Arguments to pass to VisionAssistant constructor
        
    Returns:
        VisionAssistant instance
    """
    return VisionAssistant(**kwargs)


__all__ = ['VisionAssistant', 'create_assistant']
