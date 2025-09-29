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
    
    # Model Configuration from Environment (ALL FREE MODELS)
    hf_model_primary: str = "openai/whisper-tiny"
    hf_model_backup: str = "facebook/wav2vec2-base-960h"
    groq_audio_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "llama-3.1-8b-instant"
    openrouter_model_1: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_model_2: str = "nousresearch/hermes-3-llama-3.1-405b:free"
    
    # Backward compatibility properties
    @property
    def default_hf_model(self) -> str:
        return self.hf_model_primary
    
    @property
    def backup_hf_model(self) -> str:
        return self.hf_model_backup
    
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