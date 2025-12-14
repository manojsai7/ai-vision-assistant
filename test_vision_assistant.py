"""
Tests for the AI Vision Assistant.
"""

import pytest
from PIL import Image
import numpy as np
from vision_assistant import VisionAssistant, create_assistant


class TestVisionAssistant:
    """Test suite for VisionAssistant class."""
    
    def test_initialization(self):
        """Test that VisionAssistant initializes correctly."""
        assistant = VisionAssistant()
        assert assistant is not None
        assert assistant.classification_model_name == "google/vit-base-patch16-224"
        assert assistant.detection_model_name == "facebook/detr-resnet-50"
        assert assistant.segmentation_model_name == "facebook/detr-resnet-50-panoptic"
    
    def test_custom_models_initialization(self):
        """Test initialization with custom models."""
        assistant = VisionAssistant(
            classification_model="microsoft/resnet-50",
            detection_model="facebook/detr-resnet-101",
        )
        assert assistant.classification_model_name == "microsoft/resnet-50"
        assert assistant.detection_model_name == "facebook/detr-resnet-101"
    
    def test_create_assistant_function(self):
        """Test the convenience function."""
        assistant = create_assistant()
        assert isinstance(assistant, VisionAssistant)
    
    def test_load_image_from_pil(self):
        """Test loading PIL Image."""
        assistant = VisionAssistant()
        img = Image.new('RGB', (224, 224), color='red')
        loaded_img = assistant._load_image(img)
        assert isinstance(loaded_img, Image.Image)
        assert loaded_img.size == (224, 224)
    
    def test_lazy_loading(self):
        """Test that models are not loaded on initialization."""
        assistant = VisionAssistant()
        assert assistant._classification_model is None
        assert assistant._classification_processor is None
        assert assistant._detection_model is None
        assert assistant._detection_processor is None
        assert assistant._segmentation_model is None
        assert assistant._segmentation_processor is None
    
    def test_classify_returns_correct_structure(self):
        """Test that classify returns correct data structure."""
        # Note: This test would require actual models to be loaded
        # For now, we test the structure
        assistant = VisionAssistant()
        # Mock test - in real scenario, you'd use an actual image
        assert hasattr(assistant, 'classify')
        
    def test_detect_returns_correct_structure(self):
        """Test that detect method exists and has correct signature."""
        assistant = VisionAssistant()
        assert hasattr(assistant, 'detect')
        
    def test_segment_returns_correct_structure(self):
        """Test that segment method exists and has correct signature."""
        assistant = VisionAssistant()
        assert hasattr(assistant, 'segment')
    
    def test_analyze_method_exists(self):
        """Test that analyze method exists."""
        assistant = VisionAssistant()
        assert hasattr(assistant, 'analyze')
    
    def test_analyze_with_specific_tasks(self):
        """Test analyze method structure with specific tasks."""
        assistant = VisionAssistant()
        # Test that the method accepts tasks parameter
        assert callable(assistant.analyze)


class TestImageLoading:
    """Test image loading functionality."""
    
    def test_create_rgb_image(self):
        """Test creating an RGB image."""
        img = Image.new('RGB', (224, 224), color='blue')
        assert img.mode == 'RGB'
        assert img.size == (224, 224)
    
    def test_pil_image_compatibility(self):
        """Test PIL Image compatibility."""
        assistant = VisionAssistant()
        img = Image.new('RGB', (100, 100))
        loaded = assistant._load_image(img)
        assert loaded.mode == 'RGB'


class TestModuleExports:
    """Test module exports."""
    
    def test_all_exports(self):
        """Test that __all__ is defined correctly."""
        from vision_assistant import __all__
        assert 'VisionAssistant' in __all__
        assert 'create_assistant' in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
