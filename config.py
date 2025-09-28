import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings - API Only Version."""
    
    # API Keys
    hf_api_key: str = ""
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    
    # Application Settings
    log_level: str = "INFO"
    max_file_size_mb: int = 25
    rate_limit_requests: int = 100
    rate_limit_minutes: int = 60
    
    # Audio Processing Settings (API-only, no local processing)
    sample_rate: int = 16000  # For reference only
    max_audio_duration_seconds: int = 300
    
    # Hugging Face Audio Models (API-only)
    default_hf_model: str = "superb/wav2vec2-base-superb-er"
    backup_hf_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # Groq Models (Whisper + LLM)
    groq_audio_model: str = "whisper-large-v3"
    groq_llm_model: str = "llama-3.2-11b-text-preview"
    
    # OpenRouter Models
    openrouter_model_1: str = "openai/whisper-1"
    openrouter_model_2: str = "anthropic/claude-3-haiku"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Validation
if not settings.hf_api_key:
    raise ValueError(
        "HF_API_KEY is required. Please set it in your .env file or environment variables."
    )