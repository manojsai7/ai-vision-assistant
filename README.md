# AI Vision Assistant

AI-powered computer vision assistant for image understanding, detection, and analysis.

## Features

- **Image Classification Pipeline**: Pretrained models (ResNet, MobileNet, EfficientNet) for image classification
- **Object Detection**: Faster R-CNN and RetinaNet for object detection with COCO classes
- **Image Segmentation**: DeepLabV3 and FCN for semantic segmentation
- **Pretrained Model Zoo**: Multiple pretrained models with easy selection
- **Transfer Learning Support**: Prepare models for custom tasks with configurable output classes
- **REST API**: FastAPI-based REST API for all vision tasks
- **Batch Processing**: Async job queue for processing multiple images
- **Evaluation Suite**: Standard CV metrics including mAP, IoU, F1 score, Dice coefficient
- **Docker Support**: Containerized deployment with docker-compose
- **CI/CD Ready**: GitHub Actions workflow for automated testing and deployment

## Tech Stack

- **ML Frameworks**: PyTorch, TorchVision, TensorFlow
- **API**: FastAPI, Uvicorn
- **Task Queue**: Celery, Redis
- **Computer Vision**: OpenCV, Pillow
- **Model Optimization**: ONNX, ONNX Runtime
- **Deployment**: Docker, docker-compose

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- (Optional) Docker and docker-compose

### Installation

1. Clone the repository:
```bash
git clone https://github.com/manojsai7/ai-vision-assistant
cd ai-vision-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the API server:
```bash
uvicorn app.main:app --reload
```

5. Access the API:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. Build and run with docker-compose:
```bash
docker-compose up -d
```

2. Access the API at http://localhost:8000

3. Stop the services:
```bash
docker-compose down
```

## API Endpoints

### Classification
- `POST /api/v1/classify/predict` - Classify an image
- `GET /api/v1/classify/models` - List available classification models
- `POST /api/v1/classify/transfer-learning` - Setup transfer learning

### Object Detection
- `POST /api/v1/detect/predict` - Detect objects in an image
- `GET /api/v1/detect/models` - List available detection models

### Segmentation
- `POST /api/v1/segment/predict` - Segment an image
- `GET /api/v1/segment/models` - List available segmentation models

### Evaluation
- `POST /api/v1/evaluate/iou` - Calculate IoU between bounding boxes
- `POST /api/v1/evaluate/map` - Calculate mAP for object detection
- `POST /api/v1/evaluate/f1` - Calculate F1 score for classification
- `POST /api/v1/evaluate/segmentation-metrics` - Calculate segmentation metrics
- `GET /api/v1/evaluate/metrics` - List available metrics

### Batch Processing
- `POST /api/v1/batch/submit` - Submit a batch job
- `GET /api/v1/batch/status/{job_id}` - Get job status
- `GET /api/v1/batch/jobs` - List all jobs
- `DELETE /api/v1/batch/jobs/{job_id}` - Delete a job

## Usage Examples

### Image Classification
```bash
curl -X POST "http://localhost:8000/api/v1/classify/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "model=resnet50" \
  -F "top_k=5"
```

### Object Detection
```bash
curl -X POST "http://localhost:8000/api/v1/detect/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "model=fasterrcnn_resnet50" \
  -F "confidence_threshold=0.5"
```

### Image Segmentation
```bash
curl -X POST "http://localhost:8000/api/v1/segment/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "model=deeplabv3_resnet50"
```

## Available Models

### Classification Models
- ResNet-50 (25.6M parameters)
- ResNet-18 (11.7M parameters)
- MobileNetV3-Large (5.4M parameters)
- EfficientNet-B0 (5.3M parameters)

### Detection Models
- Faster R-CNN with ResNet-50 FPN (COCO pretrained)
- RetinaNet with ResNet-50 FPN (COCO pretrained)

### Segmentation Models
- DeepLabV3 with ResNet-50 (COCO pretrained)
- FCN with ResNet-50 (COCO pretrained)

## Evaluation Metrics

- **IoU (Intersection over Union)**: For bounding box overlap
- **mAP (mean Average Precision)**: For object detection performance
- **F1 Score**: For classification performance
- **Dice Coefficient**: For segmentation overlap
- **Pixel Accuracy**: For segmentation correctness

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_api.py -v
pytest tests/test_metrics.py -v
```

## Project Structure

```
ai-vision-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/              # Model definitions
│   ├── routes/              # API routes
│   │   ├── classification.py
│   │   ├── detection.py
│   │   ├── segmentation.py
│   │   ├── evaluation.py
│   │   └── batch.py
│   ├── services/            # ML services
│   │   ├── classification.py
│   │   ├── detection.py
│   │   └── segmentation.py
│   └── utils/               # Utility functions
│       └── metrics.py
├── tests/                   # Test suite
│   ├── test_api.py
│   └── test_metrics.py
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD pipeline
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker compose setup
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Roadmap

- [ ] Add model distillation examples
- [ ] Expand pretrained backbones (Vision Transformers, YOLO)
- [ ] Add GPU-optimized Docker images
- [ ] Provide edge-device deployment guide (TensorRT, ONNX)
- [ ] Add model quantization support
- [ ] Implement gRPC API alongside REST
- [ ] Add model performance benchmarking
- [ ] Support for video processing
- [ ] Add data augmentation pipeline
- [ ] Implement model versioning and A/B testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- PyTorch and TorchVision for pretrained models
- FastAPI for the excellent web framework
- The open-source computer vision community
