"""
Tests for the main FastAPI application
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Vision Assistant"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert "features" in data


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_classification_models_endpoint():
    """Test listing classification models"""
    response = client.get("/api/v1/classify/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_detection_models_endpoint():
    """Test listing detection models"""
    response = client.get("/api/v1/detect/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_segmentation_models_endpoint():
    """Test listing segmentation models"""
    response = client.get("/api/v1/segment/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_evaluation_metrics_endpoint():
    """Test listing evaluation metrics"""
    response = client.get("/api/v1/evaluate/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert len(data["metrics"]) > 0


def test_batch_jobs_list():
    """Test listing batch jobs"""
    response = client.get("/api/v1/batch/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data
