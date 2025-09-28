import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import structlog
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from services.audio_emotion_detector import AudioEmotionDetector

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global detector instance
detector: Optional[AudioEmotionDetector] = None


# Pydantic models
class AudioEmotionRequest(BaseModel):
    """Request model for audio emotion detection."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    model_provider: str = Field("huggingface", description="AI provider to use")
    model_name: Optional[str] = Field(None, description="Specific model name")
    include_diarization: bool = Field(True, description="Include speaker diarization")
    return_segments: bool = Field(True, description="Return detailed segment information")


class EmotionResponse(BaseModel):
    """Response model for emotion detection."""
    duration_seconds: float
    sample_rate: int
    segments: List[Dict]
    overall_emotion: Optional[str]
    confidence_scores: Dict[str, float]
    processing_info: Dict
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class ModelSelectionRequest(BaseModel):
    """Request model for model selection."""
    provider: str = Field(..., description="Provider name")
    model_id: str = Field(..., description="Model identifier")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict = Field(..., description="Message data")


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection established", 
                   connections_count=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed", 
                   connections_count=len(self.active_connections))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("Error sending WebSocket message", error=str(e))
            self.disconnect(websocket)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global detector
    
    # Startup
    logger.info("Starting Audio Emotion Classifier service")
    detector = AudioEmotionDetector()
    logger.info("Audio emotion detector initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Audio Emotion Classifier service")


# FastAPI app
app = FastAPI(
    title="🎙️ Audio Emotion Classifier Pro",
    description="Real-time AI-Powered Audio Emotion Recognition with Speaker Diarization",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Serve static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application."""
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <head><title>Audio Emotion Classifier</title></head>
            <body>
                <h1>🎙️ Audio Emotion Classifier Pro</h1>
                <p>Static files not found. Please ensure the frontend is built.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "audio-emotion-classifier",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models_available": {
            "huggingface": bool(settings.hf_api_key),
            "groq": bool(settings.groq_api_key),
            "openrouter": bool(settings.openrouter_api_key)
        }
    }


@app.post("/api/detect", response_model=EmotionResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_minutes}minute")
async def detect_emotion(
    request: AudioEmotionRequest,
    http_request=None
) -> EmotionResponse:
    """Detect emotions from audio with speaker diarization."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    start_time = time.time()
    
    try:
        logger.info("Processing audio emotion detection request",
                   provider=request.model_provider,
                   model=request.model_name,
                   diarization=request.include_diarization,
                   api_only=True)
        
        result = await detector.process_audio(
            audio_data=request.audio_data,
            model_provider=request.model_provider,
            model_name=request.model_name,
            include_diarization=request.include_diarization,
            return_segments=request.return_segments
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        logger.info("Audio emotion detection completed",
                   processing_time_ms=processing_time,
                   segments_count=len(result.get("segments", [])),
                   overall_emotion=result.get("overall_emotion"))
        
        return EmotionResponse(**result)
        
    except Exception as e:
        logger.error("Error in emotion detection", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/file")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_minutes}minute")
async def detect_emotion_from_file(
    http_request=None,
    file: UploadFile = File(...),
    model_provider: str = Form("huggingface"),
    model_name: Optional[str] = Form(None),
    include_diarization: bool = Form(True),
    return_segments: bool = Form(True)
) -> EmotionResponse:
    """Detect emotions from uploaded audio file."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    # Validate file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    start_time = time.time()
    
    try:
        logger.info("Processing uploaded audio file",
                   filename=file.filename,
                   size_mb=file_size_mb,
                   provider=model_provider,
                   api_only=True)
        
        result = await detector.process_audio(
            audio_data=content,
            model_provider=model_provider,
            model_name=model_name,
            include_diarization=include_diarization,
            return_segments=return_segments
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        logger.info("File emotion detection completed",
                   filename=file.filename,
                   processing_time_ms=processing_time)
        
        return EmotionResponse(**result)
        
    except Exception as e:
        logger.error("Error processing uploaded file", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_available_models():
    """Get available models for each provider."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        models = await detector.get_available_models()
        return {
            "available_models": models,
            "default_provider": "huggingface",
            "providers_status": {
                "huggingface": bool(settings.hf_api_key),
                "groq": bool(settings.groq_api_key),
                "openrouter": bool(settings.openrouter_api_key)
            }
        }
    except Exception as e:
        logger.error("Error getting available models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/select")
async def select_model(request: ModelSelectionRequest):
    """Select a specific model (for future use)."""
    return {
        "message": f"Model selection noted: {request.provider}/{request.model_id}",
        "provider": request.provider,
        "model_id": request.model_id,
        "status": "acknowledged"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio processing."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio_chunk":
                # Process audio chunk
                chunk_data = message.get("data", {})
                audio_data = chunk_data.get("audio_data")
                
                if audio_data and detector:
                    try:
                        result = await detector.process_audio(
                            audio_data=audio_data,
                            model_provider=chunk_data.get("model_provider", "huggingface"),
                            model_name=chunk_data.get("model_name"),
                            include_diarization=chunk_data.get("include_diarization", True),
                            return_segments=True
                        )
                        
                        response = {
                            "type": "emotion_result",
                            "data": result,
                            "timestamp": time.time()
                        }
                        
                        await manager.send_personal_message(
                            json.dumps(response), websocket
                        )
                        
                    except Exception as e:
                        error_response = {
                            "type": "error",
                            "data": {"message": str(e)},
                            "timestamp": time.time()
                        }
                        await manager.send_personal_message(
                            json.dumps(error_response), websocket
                        )
            
            elif message.get("type") == "ping":
                # Respond to ping
                pong_response = {
                    "type": "pong",
                    "timestamp": time.time()
                }
                await manager.send_personal_message(
                    json.dumps(pong_response), websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics."""
    import psutil
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "connections": {
            "websocket_connections": len(manager.active_connections)
        },
        "settings": {
            "max_file_size_mb": settings.max_file_size_mb,
            "sample_rate": settings.sample_rate,
            "max_audio_duration": settings.max_audio_duration_seconds
        }
    }


if __name__ == "__main__":
    import sys
    
    # Configure logging level
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("Starting Audio Emotion Classifier server",
               port=port,
               log_level=settings.log_level)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload="--reload" in sys.argv,
        access_log=True
    )