"""
AI Vision Assistant - Main FastAPI Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.routes import classification, detection, segmentation, evaluation, batch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting AI Vision Assistant API...")
    logger.info("Loading pretrained models...")
    yield
    # Shutdown
    logger.info("Shutting down AI Vision Assistant API...")


# Initialize FastAPI app
app = FastAPI(
    title="AI Vision Assistant",
    description="AI-powered computer vision assistant for image understanding, detection, and analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classification.router, prefix="/api/v1/classify", tags=["Classification"])
app.include_router(detection.router, prefix="/api/v1/detect", tags=["Detection"])
app.include_router(segmentation.router, prefix="/api/v1/segment", tags=["Segmentation"])
app.include_router(evaluation.router, prefix="/api/v1/evaluate", tags=["Evaluation"])
app.include_router(batch.router, prefix="/api/v1/batch", tags=["Batch Processing"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Vision Assistant",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Image Classification",
            "Object Detection",
            "Image Segmentation",
            "Batch Processing",
            "Model Evaluation",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
