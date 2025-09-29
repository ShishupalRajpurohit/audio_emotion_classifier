import asyncio
import base64
import io
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import httpx
from config import settings

logger = logging.getLogger(__name__)


class AudioEmotionDetector:
    """API-only Audio emotion detection - analyzes VOICE TONE, not text sentiment."""
    
    def __init__(self):
        self.hf_headers = {"Authorization": f"Bearer {settings.hf_api_key}"}
        self.groq_headers = {"Authorization": f"Bearer {settings.groq_api_key}"} if settings.groq_api_key else None
        self.openrouter_headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"} if settings.openrouter_api_key else None
        
        # Emotion mapping
        self.emotion_labels = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

    async def process_audio(
        self,
        audio_data: Union[bytes, str],
        model_provider: str = "huggingface",  # HF for voice tone analysis
        model_name: Optional[str] = None,
        include_diarization: bool = True,
        return_segments: bool = True
    ) -> Dict:
        """
        Process audio: Get transcription + analyze VOICE TONE for emotion.
        """
        try:
            # Convert audio data
            audio_bytes = await self._prepare_audio_bytes(audio_data)
            
            # Validate file size
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                raise ValueError(f"Audio too large: {file_size_mb:.1f}MB")
            
            results = {
                "duration_seconds": 0.0,
                "sample_rate": settings.sample_rate,
                "segments": [],
                "overall_emotion": None,
                "confidence_scores": {},
                "transcription": "",
                "labeled_transcription": "",
                "processing_info": {
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "api_only": True
                }
            }
            
            # Step 1: Get transcription (Groq Whisper)
            transcription_data = await self._get_transcription(audio_bytes)
            results["transcription"] = transcription_data.get("text", "")
            results["duration_seconds"] = transcription_data.get("duration", self._estimate_duration(audio_bytes))
            
            # Step 2: Analyze VOICE TONE emotion from audio (HuggingFace audio emotion model)
            voice_emotion = await self._analyze_voice_emotion(audio_bytes, model_provider, model_name)
            
            # Step 3: Create segments based on transcription timing
            if include_diarization and results["transcription"]:
                segments = await self._create_voice_emotion_segments(
                    transcription_data, voice_emotion, results["duration_seconds"]
                )
            else:
                # Single segment with voice emotion
                segments = [{
                    "start_ms": 0,
                    "end_ms": int(results["duration_seconds"] * 1000),
                    "speaker": "speaker_0",
                    "text": results["transcription"],
                    "emotion": voice_emotion["emotion"],
                    "confidence": voice_emotion["confidence"],
                    "all_scores": voice_emotion["all_scores"],
                    "duration_ms": int(results["duration_seconds"] * 1000)
                }]
            
            results["segments"] = segments
            results["overall_emotion"] = voice_emotion["emotion"]
            results["confidence_scores"] = voice_emotion["all_scores"]
            
            # Create labeled transcription with voice emotion
            if results["transcription"]:
                emotion = voice_emotion["emotion"]
                results["labeled_transcription"] = f"[{emotion}]{results['transcription']}"
            
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
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            return audio_bytes
        except Exception as e:
            logger.error(f"Error preparing audio: {str(e)}")
            raise ValueError(f"Invalid audio format: {str(e)}")

    async def _get_transcription(self, audio_bytes: bytes) -> Dict:
        """Get transcription from Groq Whisper."""
        if not self.groq_headers:
            return {"text": "", "duration": self._estimate_duration(audio_bytes), "words": [], "segments": []}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data = {"model": settings.groq_audio_model}
                
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=self.groq_headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "")
                    duration = result.get("duration", self._estimate_duration(audio_bytes))
                    
                    return {
                        "text": text,
                        "duration": duration,
                        "words": result.get("words", []),
                        "segments": result.get("segments", [])
                    }
                else:
                    logger.warning(f"Groq transcription error {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Groq transcription error: {str(e)}")
        
        return {"text": "", "duration": self._estimate_duration(audio_bytes), "words": [], "segments": []}

    async def _analyze_voice_emotion(self, audio_bytes: bytes, provider: str, model_name: Optional[str]) -> Dict:
        """Analyze emotion from VOICE TONE/PROSODY using audio emotion model."""
        # Always use HuggingFace audio emotion model for voice analysis
        return await self._hf_audio_emotion(audio_bytes, model_name)

    async def _hf_audio_emotion(self, audio_bytes: bytes, model_name: Optional[str]) -> Dict:
        """Analyze VOICE EMOTION using HuggingFace audio emotion recognition model."""
        model = model_name or settings.hf_audio_emotion_model
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers=self.hf_headers,
                    data=audio_bytes
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle response format
                    if isinstance(result, list) and result:
                        # Standard format: [{"label": "emotion", "score": 0.x}]
                        scores = {}
                        for item in result:
                            label = item.get('label', '').lower()
                            score = item.get('score', 0)
                            
                            # Normalize emotion labels
                            if 'ang' in label:
                                scores['angry'] = score
                            elif 'hap' in label or 'joy' in label:
                                scores['happy'] = score
                            elif 'sad' in label:
                                scores['sad'] = score
                            elif 'fear' in label:
                                scores['fear'] = score
                            elif 'surp' in label:
                                scores['surprise'] = score
                            elif 'neut' in label or 'calm' in label:
                                scores['neutral'] = score
                            elif 'disg' in label:
                                scores['disgust'] = score
                            else:
                                # Keep original label if no match
                                scores[label] = score
                        
                        if scores:
                            best_emotion = max(scores.items(), key=lambda x: x[1])
                            
                            return {
                                "emotion": best_emotion[0],
                                "confidence": best_emotion[1],
                                "all_scores": scores,
                                "model_used": model,
                                "source": "voice_tone"
                            }
                    
                    # Try backup model
                    if model != settings.hf_audio_emotion_backup:
                        logger.info(f"Trying backup model: {settings.hf_audio_emotion_backup}")
                        return await self._hf_audio_emotion(audio_bytes, settings.hf_audio_emotion_backup)
                        
                else:
                    logger.warning(f"HF API error {response.status_code}: {response.text}")
                    # Try backup
                    if model != settings.hf_audio_emotion_backup:
                        return await self._hf_audio_emotion(audio_bytes, settings.hf_audio_emotion_backup)
                    
        except Exception as e:
            logger.error(f"HF audio emotion error: {str(e)}")
            
        # Fallback
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "all_scores": {"neutral": 0.5},
            "model_used": model,
            "fallback": True,
            "source": "fallback"
        }

    async def _create_voice_emotion_segments(
        self, 
        transcription_data: Dict, 
        voice_emotion: Dict,
        duration: float
    ) -> List[Dict]:
        """Create segments based on sentences, with VOICE emotion applied to all."""
        try:
            text = transcription_data.get("text", "")
            
            if not text.strip():
                return []
            
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return []
            
            segments = []
            sentence_duration = duration / len(sentences)
            current_time = 0.0
            
            for i, sentence in enumerate(sentences):
                start_ms = int(current_time * 1000)
                end_ms = int((current_time + sentence_duration) * 1000)
                
                # Use VOICE emotion for all segments (not text analysis)
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": f"speaker_{i % 2}",
                    "text": sentence.strip(),
                    "duration_ms": end_ms - start_ms,
                    "emotion": voice_emotion["emotion"],
                    "confidence": voice_emotion["confidence"],
                    "all_scores": voice_emotion["all_scores"]
                })
                
                current_time += sentence_duration
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating segments: {str(e)}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s*|\s*\.\.\.\s*', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if len(sentences) < 2 and len(text) > 50:
            sentences = re.split(r',\s+|\s+(?:and|but|however)\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences[:10]

    def _estimate_duration(self, audio_bytes: bytes) -> float:
        """Estimate duration from audio file size."""
        estimated = len(audio_bytes) / (16000 * 2)
        return max(0.5, min(estimated, 300))

    async def get_available_models(self) -> Dict:
        """Get available models."""
        return {
            "huggingface": [
                {
                    "id": settings.hf_audio_emotion_model,
                    "name": "Wav2Vec2 Audio Emotion (Voice Tone)",
                    "description": "Analyzes emotion from voice prosody/tone"
                },
                {
                    "id": settings.hf_audio_emotion_backup,
                    "name": "HuBERT Audio Emotion (Voice Tone)",
                    "description": "Alternative voice emotion model"
                }
            ],
            "groq": [
                {
                    "id": "whisper_transcription",
                    "name": "Whisper Transcription Only",
                    "description": "Transcribes speech to text (no emotion)"
                }
            ] if self.groq_headers else [],
            "openrouter": []  # Not used for audio emotion
        }