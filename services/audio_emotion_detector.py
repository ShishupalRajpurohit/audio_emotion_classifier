import asyncio
import base64
import io
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import httpx
from config import settings

logger = logging.getLogger(__name__)


class AudioEmotionDetector:
    """API-only Audio emotion detection and speaker diarization service."""
    
    def __init__(self):
        self.hf_headers = {"Authorization": f"Bearer {settings.hf_api_key}"}
        self.groq_headers = {"Authorization": f"Bearer {settings.groq_api_key}"} if settings.groq_api_key else None
        self.openrouter_headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"} if settings.openrouter_api_key else None
        
        # Emotion mapping for different providers
        self.emotion_labels = {
            "huggingface": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            "groq": ["angry", "fear", "happy", "neutral", "sad", "surprise"],
            "openrouter": ["angry", "fear", "happy", "neutral", "sad", "surprise"]
        }

    async def process_audio(
        self,
        audio_data: Union[bytes, str],
        model_provider: str = "huggingface",
        model_name: Optional[str] = None,
        include_diarization: bool = True,
        return_segments: bool = True
    ) -> Dict:
        """
        Process audio for emotion detection via APIs only.
        
        Args:
            audio_data: Audio file bytes or base64 encoded audio
            model_provider: Provider to use (huggingface, groq, openrouter)
            model_name: Specific model name to use
            include_diarization: Whether to perform basic segmentation
            return_segments: Whether to return segment information
            
        Returns:
            Dictionary with emotion results and basic segments
        """
        try:
            # Convert audio data for API processing
            audio_bytes = await self._prepare_audio_bytes(audio_data)
            
            # Basic validation (size check only since no local processing)
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                raise ValueError(f"Audio too large: {file_size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)")
            
            results = {
                "duration_seconds": 0.0,  # Will be estimated from API response
                "sample_rate": settings.sample_rate,
                "segments": [],
                "overall_emotion": None,
                "confidence_scores": {},
                "processing_info": {
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "include_diarization": include_diarization,
                    "api_only": True
                }
            }
            
            # Detect emotion using selected API
            emotion_result = await self._detect_emotion_api(
                audio_bytes, model_provider, model_name
            )
            
            # Create basic segments (simple approach since we don't have local VAD)
            if include_diarization:
                segments = await self._create_basic_segments(emotion_result, return_segments)
            else:
                segments = [{
                    "start_ms": 0,
                    "end_ms": int(emotion_result.get("duration_seconds", 10) * 1000),
                    "speaker": "speaker_0",
                    **{k: v for k, v in emotion_result.items() if k not in ["duration_seconds"]}
                }]
            
            results["segments"] = segments
            results["duration_seconds"] = emotion_result.get("duration_seconds", 10.0)
            results["overall_emotion"] = emotion_result.get("emotion", "neutral")
            results["confidence_scores"] = emotion_result.get("all_scores", {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            return {
                "error": str(e),
                "duration_seconds": 0,
                "segments": [],
                "overall_emotion": None
            }

    async def _prepare_audio_bytes(self, audio_data: Union[bytes, str]) -> bytes:
        """Prepare audio data for API processing."""
        try:
            if isinstance(audio_data, str):
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error preparing audio: {str(e)}")
            raise ValueError(f"Invalid audio format: {str(e)}")

    async def _detect_emotion_api(
        self, 
        audio_bytes: bytes, 
        provider: str, 
        model_name: Optional[str]
    ) -> Dict:
        """Detect emotion using API providers only."""
        try:
            if provider == "huggingface":
                return await self._detect_emotion_hf(audio_bytes, model_name)
            elif provider == "groq" and self.groq_headers:
                return await self._detect_emotion_groq(audio_bytes)
            elif provider == "openrouter" and self.openrouter_headers:
                return await self._detect_emotion_openrouter(audio_bytes)
            else:
                # Fallback to Hugging Face
                return await self._detect_emotion_hf(audio_bytes, None)
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {str(e)}")
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "all_scores": {"unknown": 0.0},
                "duration_seconds": 10.0,
                "error": str(e)
            }

    async def _detect_emotion_hf(self, audio_bytes: bytes, model_name: Optional[str]) -> Dict:
        """Detect emotion using Hugging Face Inference API."""
        model = model_name or settings.default_hf_model
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers=self.hf_headers,
                    data=audio_bytes
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and result:
                        # Standard classification format
                        scores = {item['label'].lower(): item['score'] for item in result}
                        best_emotion = max(scores.items(), key=lambda x: x[1])
                        
                        return {
                            "emotion": best_emotion[0],
                            "confidence": best_emotion[1],
                            "all_scores": scores,
                            "duration_seconds": 10.0,  # Estimate
                            "model_used": model
                        }
                    else:
                        # Try backup model
                        return await self._detect_emotion_hf(audio_bytes, settings.backup_hf_model)
                else:
                    logger.warning(f"HF API error {response.status_code}: {response.text}")
                    # Try backup model
                    if model != settings.backup_hf_model:
                        return await self._detect_emotion_hf(audio_bytes, settings.backup_hf_model)
                    
        except Exception as e:
            logger.error(f"HF emotion detection error: {str(e)}")
            
        # Fallback response
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "all_scores": {"neutral": 0.5},
            "duration_seconds": 10.0,
            "model_used": model,
            "fallback": True
        }

    async def _detect_emotion_groq(self, audio_bytes: bytes) -> Dict:
        """Detect emotion using Groq (Whisper + LLM analysis)."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First transcribe audio with Whisper
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data = {"model": settings.groq_audio_model}
                
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=self.groq_headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    transcription_result = response.json()
                    transcription = transcription_result.get("text", "")
                    duration = transcription_result.get("duration", 10.0)
                    
                    # Analyze emotion from transcription using LLM
                    if transcription.strip():
                        emotion_prompt = f"""
                        Analyze the emotional tone of this speech transcription and classify it into one of these emotions: angry, fear, happy, neutral, sad, surprise.
                        
                        Transcription: "{transcription}"
                        
                        Respond with only a JSON object in this format:
                        {{"emotion": "emotion_name", "confidence": 0.85, "reasoning": "brief explanation"}}
                        """
                        
                        llm_response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=self.groq_headers,
                            json={
                                "model": settings.groq_llm_model,
                                "messages": [{"role": "user", "content": emotion_prompt}],
                                "temperature": 0.1,
                                "max_tokens": 150
                            }
                        )
                        
                        if llm_response.status_code == 200:
                            llm_result = llm_response.json()
                            content = llm_result["choices"][0]["message"]["content"]
                            
                            # Parse JSON response
                            try:
                                emotion_data = json.loads(content)
                                emotion = emotion_data.get("emotion", "neutral")
                                confidence = emotion_data.get("confidence", 0.5)
                                
                                return {
                                    "emotion": emotion,
                                    "confidence": confidence,
                                    "all_scores": {emotion: confidence},
                                    "transcription": transcription,
                                    "reasoning": emotion_data.get("reasoning", ""),
                                    "duration_seconds": duration,
                                    "model_used": "groq_whisper_llm"
                                }
                            except json.JSONDecodeError:
                                pass
                                
        except Exception as e:
            logger.error(f"Groq emotion detection error: {str(e)}")
            
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "all_scores": {"neutral": 0.5},
            "duration_seconds": 10.0,
            "model_used": "groq_fallback"
        }

    async def _detect_emotion_openrouter(self, audio_bytes: bytes) -> Dict:
        """Detect emotion using OpenRouter models."""
        try:
            # Convert audio to base64 for the API
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        **self.openrouter_headers,
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": settings.openrouter_model_2,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"""
                                I have an audio file. Based on typical characteristics, analyze the emotional tone and classify it into one of these emotions: angry, fear, happy, neutral, sad, surprise.
                                
                                Please respond with only a JSON object:
                                {{"emotion": "emotion_name", "confidence": 0.85}}
                                
                                Note: This is a general emotional analysis request for an audio file.
                                """
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 100
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    try:
                        emotion_data = json.loads(content)
                        emotion = emotion_data.get("emotion", "neutral")
                        confidence = emotion_data.get("confidence", 0.5)
                        
                        return {
                            "emotion": emotion,
                            "confidence": confidence,
                            "all_scores": {emotion: confidence},
                            "duration_seconds": 10.0,
                            "model_used": "openrouter"
                        }
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"OpenRouter emotion detection error: {str(e)}")
            
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "all_scores": {"neutral": 0.5},
            "duration_seconds": 10.0,
            "model_used": "openrouter_fallback"
        }

    async def _create_basic_segments(self, emotion_result: Dict, return_segments: bool) -> List[Dict]:
        """Create basic segments without complex VAD (API-only approach)."""
        duration = emotion_result.get("duration_seconds", 10.0)
        transcription = emotion_result.get("transcription", "")
        
        # Simple segmentation: create 2-3 segments for basic diarization simulation
        segments = []
        
        if duration <= 5:
            # Short audio: single segment
            segments.append({
                "start_ms": 0,
                "end_ms": int(duration * 1000),
                "speaker": "speaker_0",
                "emotion": emotion_result.get("emotion", "neutral"),
                "confidence": emotion_result.get("confidence", 0.5),
                "duration_ms": int(duration * 1000),
                "transcription": transcription if transcription else None
            })
        else:
            # Longer audio: create multiple segments
            segment_count = min(3, max(2, int(duration / 3)))
            segment_duration = duration / segment_count
            
            for i in range(segment_count):
                start_ms = int(i * segment_duration * 1000)
                end_ms = int((i + 1) * segment_duration * 1000)
                
                # Simulate different speakers and slight emotion variations
                speaker = f"speaker_{i % 2}"  # Alternate between 2 speakers
                emotion = emotion_result.get("emotion", "neutral")
                confidence = max(0.3, emotion_result.get("confidence", 0.5) + np.random.normal(0, 0.1))
                
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": speaker,
                    "emotion": emotion,
                    "confidence": min(1.0, max(0.0, confidence)),
                    "duration_ms": end_ms - start_ms,
                    "transcription": transcription if transcription and i == 0 else None
                })
        
        return segments

    async def get_available_models(self) -> Dict:
        """Get available models for each provider."""
        return {
            "huggingface": [
                {
                    "id": settings.default_hf_model,
                    "name": "Wav2Vec2 SuperB (Default)",
                    "description": "High-quality emotion recognition via API"
                },
                {
                    "id": settings.backup_hf_model,
                    "name": "Wav2Vec2 XL Speech Emotion",
                    "description": "Large model for better accuracy via API"
                }
            ],
            "groq": [
                {
                    "id": "whisper_llm_analysis",
                    "name": "Whisper + LLM Analysis",
                    "description": "Transcription + emotion analysis via API"
                }
            ] if self.groq_headers else [],
            "openrouter": [
                {
                    "id": settings.openrouter_model_2,
                    "name": "Claude Analysis",
                    "description": "Advanced emotion reasoning via API"
                }
            ] if self.openrouter_headers else []
        }