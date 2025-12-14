# AI Vision Assistant 🤖👁️

An AI sidekick that understands images—classifies, detects, and segments—ready to plug into your apps.

## Features ✨

- **Image Classification**: Identify what's in an image with state-of-the-art deep learning models
- **Object Detection**: Find and locate multiple objects within images with bounding boxes
- **Image Segmentation**: Separate and identify different regions/objects in images
- **Easy Integration**: Simple API designed to plug into your applications
- **Pre-trained Models**: Uses powerful pre-trained models from HuggingFace
- **Flexible**: Supports custom models and configurations

## Installation 📦

```bash
pip install -r requirements.txt
```

## Quick Start 🚀

```python
from vision_assistant import VisionAssistant

# Create an assistant
assistant = VisionAssistant()

# Classify an image
results = assistant.classify("path/to/image.jpg", top_k=5)
print(results)
# Output: [{'label': 'golden retriever', 'score': 0.95}, ...]

# Detect objects
detections = assistant.detect("path/to/scene.jpg")
for obj in detections:
    print(f"{obj['label']}: {obj['score']:.2f} at {obj['box']}")
# Output: person: 0.98 at [100.5, 50.3, 300.7, 400.2]

# Segment image
segments = assistant.segment("path/to/image.jpg")
print(f"Found {len(segments['segments'])} segments")

# Analyze with all tasks at once
results = assistant.analyze("path/to/image.jpg")
print(results['classification'])
print(results['detection'])
print(results['segmentation'])
```

## API Reference 📚

### VisionAssistant

The main class that provides all vision capabilities.

#### `__init__(classification_model, detection_model, segmentation_model)`

Initialize the vision assistant with optional custom models.

**Parameters:**
- `classification_model` (str): HuggingFace model for classification (default: "google/vit-base-patch16-224")
- `detection_model` (str): HuggingFace model for object detection (default: "facebook/detr-resnet-50")
- `segmentation_model` (str): HuggingFace model for segmentation (default: "facebook/detr-resnet-50-panoptic")

#### `classify(image, top_k=5)`

Classify an image and return top predictions.

**Parameters:**
- `image` (str or PIL.Image): Path to image or PIL Image object
- `top_k` (int): Number of top predictions to return

**Returns:**
- List of dicts with `label` and `score` keys

#### `detect(image, threshold=0.9)`

Detect objects in an image.

**Parameters:**
- `image` (str or PIL.Image): Path to image or PIL Image object
- `threshold` (float): Confidence threshold for detections

**Returns:**
- List of dicts with `label`, `score`, and `box` keys

#### `segment(image, threshold=0.9)`

Perform panoptic segmentation on an image.

**Parameters:**
- `image` (str or PIL.Image): Path to image or PIL Image object
- `threshold` (float): Confidence threshold for segments

**Returns:**
- Dict with `segments` list and `segmentation_map` array

#### `analyze(image, tasks=None)`

Run multiple vision tasks on a single image.

**Parameters:**
- `image` (str or PIL.Image): Path to image or PIL Image object
- `tasks` (list): List of tasks ['classify', 'detect', 'segment'] (default: all)

**Returns:**
- Dict with results for each requested task

## Examples 💡

See `examples.py` for detailed usage examples:

```bash
python examples.py
```

## Use Cases 🎯

- **Content Moderation**: Automatically classify and detect inappropriate content
- **E-commerce**: Identify products and extract features from images
- **Autonomous Systems**: Object detection for robotics and self-driving cars
- **Medical Imaging**: Segment and classify medical images
- **Smart Home**: Recognize objects and people for security systems
- **Image Search**: Build visual search engines with classification
- **Accessibility**: Generate image descriptions for visually impaired users

## Custom Models 🔧

You can use any compatible model from HuggingFace:

```python
assistant = VisionAssistant(
    classification_model="microsoft/resnet-50",
    detection_model="facebook/detr-resnet-101",
    segmentation_model="nvidia/segformer-b5-finetuned-ade-640-640"
)
```

## Architecture 🏗️

The assistant uses:
- **Vision Transformers (ViT)** for image classification
- **DETR (Detection Transformer)** for object detection
- **Panoptic Segmentation** for image segmentation
- **PyTorch** as the deep learning framework
- **HuggingFace Transformers** for pre-trained models

## Requirements 📋

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- Pillow 10.0+
- NumPy 1.24+

See `requirements.txt` for complete dependencies.

## Performance Tips ⚡

- Models are loaded lazily on first use to save memory
- Use GPU for faster inference: `torch.cuda.is_available()`
- Cache models locally to avoid re-downloading
- Batch process multiple images when possible

## Contributing 🤝

Contributions are welcome! This project provides a foundation for computer vision tasks that can be extended with:
- Additional model support
- Batch processing capabilities
- Video analysis features
- More segmentation methods
- Custom training pipelines

## License 📄

MIT License - feel free to use in your projects!

## Acknowledgments 🙏

Built with amazing open-source tools:
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [DETR](https://github.com/facebookresearch/detr)
- [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer)
