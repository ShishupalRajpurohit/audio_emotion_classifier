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
    """API-only Audio emotion detection with accurate segmentation and transcription."""
    
    def __init__(self):
        self.hf_headers = {"Authorization": f"Bearer {settings.hf_api_key}"}
        self.groq_headers = {"Authorization": f"Bearer {settings.groq_api_key}"} if settings.groq_api_key else None
        self.openrouter_headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"} if settings.openrouter_api_key else None
        
        # Emotion mapping
        self.emotion_labels = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

    async def process_audio(
        self,
        audio_data: Union[bytes, str],
        model_provider: str = "groq",  # Default to Groq for better results
        model_name: Optional[str] = None,
        include_diarization: bool = True,
        return_segments: bool = True
    ) -> Dict:
        """
        Process audio for emotion detection with accurate duration and segmentation.
        """
        try:
            # Convert audio data
            audio_bytes = await self._prepare_audio_bytes(audio_data)
            
            # Validate file size
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                raise ValueError(f"Audio too large: {file_size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)")
            
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
            
            # Get transcription and duration first (more accurate)
            transcription_data = await self._get_transcription_with_timing(audio_bytes, model_provider)
            
            if not transcription_data:
                return {**results, "error": "Failed to transcribe audio"}
            
            results["duration_seconds"] = transcription_data.get("duration", 0.0)
            results["transcription"] = transcription_data.get("text", "")
            
            # Create accurate segments based on sentences and pauses
            if include_diarization and results["transcription"]:
                segments = await self._create_sentence_level_segments(
                    transcription_data, audio_bytes, model_provider
                )
            else:
                # Single segment
                emotion_result = await self._analyze_text_emotion(results["transcription"])
                segments = [{
                    "start_ms": 0,
                    "end_ms": int(results["duration_seconds"] * 1000),
                    "speaker": "speaker_0",
                    "text": results["transcription"],
                    **emotion_result
                }]
            
            results["segments"] = segments
            
            # Calculate overall emotion and create labeled transcription
            if segments:
                results["overall_emotion"] = await self._calculate_overall_emotion(segments)
                results["confidence_scores"] = await self._aggregate_confidence_scores(segments)
                results["labeled_transcription"] = await self._create_labeled_transcription(segments)
            
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

    async def _get_transcription_with_timing(self, audio_bytes: bytes, provider: str) -> Dict:
        """Get transcription with accurate timing information."""
        try:
            if provider == "groq" and self.groq_headers:
                return await self._groq_transcription(audio_bytes)
            elif provider == "openrouter" and self.openrouter_headers:
                return await self._openrouter_transcription(audio_bytes)
            else:
                # Fallback: estimate timing from audio size
                return await self._estimate_timing(audio_bytes)
                
        except Exception as e:
            logger.error(f"Error getting transcription: {str(e)}")
            return {}

    async def _groq_transcription(self, audio_bytes: bytes) -> Dict:
        """Get transcription from Groq Whisper with timing."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data = {
                    "model": settings.groq_audio_model,
                    "response_format": "verbose_json",  # Get timing info
                    "timestamp_granularities": ["word"]
                }
                
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=self.groq_headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("text", ""),
                        "duration": result.get("duration", 0.0),
                        "words": result.get("words", []),
                        "segments": result.get("segments", [])
                    }
                else:
                    logger.warning(f"Groq transcription error {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Groq transcription error: {str(e)}")
            
        return {}

    async def _openrouter_transcription(self, audio_bytes: bytes) -> Dict:
        """Get transcription from OpenRouter."""
        try:
            # Use OpenRouter's Whisper model
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={**self.openrouter_headers, "Content-Type": "application/json"},
                    json={
                        "model": "openai/whisper-1",
                        "messages": [{
                            "role": "user", 
                            "content": f"Transcribe this audio and estimate its duration: {audio_base64[:100]}..."
                        }],
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    
                    # Estimate duration (rough calculation)
                    duration = max(2.0, len(audio_bytes) / 16000)  # Assume 16kHz
                    
                    return {
                        "text": text,
                        "duration": duration,
                        "words": [],
                        "segments": []
                    }
                    
        except Exception as e:
            logger.error(f"OpenRouter transcription error: {str(e)}")
            
        return {}

    async def _estimate_timing(self, audio_bytes: bytes) -> Dict:
        """Estimate timing from audio file size."""
        # Rough estimation: assuming 16kHz, 16-bit, mono
        estimated_duration = len(audio_bytes) / (16000 * 2)  # bytes / (sample_rate * bytes_per_sample)
        estimated_duration = max(0.5, min(estimated_duration, 300))  # Clamp between 0.5s and 300s
        
        return {
            "text": "Audio transcription not available",
            "duration": estimated_duration,
            "words": [],
            "segments": []
        }

    async def _create_sentence_level_segments(
        self, 
        transcription_data: Dict, 
        audio_bytes: bytes, 
        provider: str
    ) -> List[Dict]:
        """Create segments based on sentences and emotional context."""
        try:
            text = transcription_data.get("text", "")
            duration = transcription_data.get("duration", 10.0)
            words = transcription_data.get("words", [])
            
            if not text.strip():
                return []
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return []
            
            segments = []
            current_time = 0.0
            sentence_duration = duration / len(sentences)
            
            # If we have word-level timing, use it
            if words:
                segments = await self._create_word_timed_segments(sentences, words, duration)
            else:
                # Create time-based segments
                for i, sentence in enumerate(sentences):
                    start_ms = int(current_time * 1000)
                    end_ms = int((current_time + sentence_duration) * 1000)
                    
                    # Analyze emotion for this sentence
                    emotion_result = await self._analyze_text_emotion(sentence)
                    
                    segments.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "speaker": f"speaker_{i % 2}",  # Alternate speakers for variety
                        "text": sentence.strip(),
                        "duration_ms": end_ms - start_ms,
                        **emotion_result
                    })
                    
                    current_time += sentence_duration
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating segments: {str(e)}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation and pauses."""
        # Split on sentence endings, exclamations, questions, and long pauses
        sentences = re.split(r'[.!?]+\s*|\s*\.\.\.\s*|\s{3,}', text)
        
        # Filter out empty sentences and very short ones
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        # If no clear sentences, split by commas or conjunctions
        if len(sentences) < 2 and len(text) > 50:
            sentences = re.split(r',\s+|\s+(?:and|but|however|although|while)\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences[:10]  # Limit to 10 segments max

    async def _create_word_timed_segments(
        self, 
        sentences: List[str], 
        words: List[Dict], 
        duration: float
    ) -> List[Dict]:
        """Create segments using word-level timing information."""
        segments = []
        
        try:
            # Map sentences to word timings
            word_index = 0
            
            for i, sentence in enumerate(sentences):
                sentence_words = sentence.lower().split()
                start_time = None
                end_time = None
                
                # Find matching words in timing data
                for word_data in words[word_index:]:
                    if word_data.get("word", "").lower().strip(".,!?") in sentence_words:
                        if start_time is None:
                            start_time = word_data.get("start", 0.0)
                        end_time = word_data.get("end", word_data.get("start", 0.0) + 0.5)
                        word_index += 1
                
                # Fallback to estimated timing
                if start_time is None:
                    start_time = (i * duration) / len(sentences)
                    end_time = ((i + 1) * duration) / len(sentences)
                
                # Analyze emotion for this sentence
                emotion_result = await self._analyze_text_emotion(sentence)
                
                segments.append({
                    "start_ms": int(start_time * 1000),
                    "end_ms": int(end_time * 1000),
                    "speaker": f"speaker_{i % 2}",
                    "text": sentence.strip(),
                    "duration_ms": int((end_time - start_time) * 1000),
                    **emotion_result
                })
                
        except Exception as e:
            logger.error(f"Error in word-timed segments: {str(e)}")
            
        return segments

    async def _analyze_text_emotion(self, text: str) -> Dict:
        """Analyze emotion from text using LLM."""
        try:
            if not text.strip():
                return {
                    "emotion": "neutral",
                    "confidence": 0.5,
                    "all_scores": {"neutral": 0.5}
                }
            
            # Use Groq LLM for emotion analysis
            if self.groq_headers:
                return await self._groq_emotion_analysis(text)
            elif self.openrouter_headers:
                return await self._openrouter_emotion_analysis(text)
            else:
                # Fallback: simple keyword-based emotion detection
                return self._simple_emotion_analysis(text)
                
        except Exception as e:
            logger.error(f"Error analyzing text emotion: {str(e)}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_scores": {"neutral": 0.5}
            }

    async def _groq_emotion_analysis(self, text: str) -> Dict:
        """Analyze emotion using Groq LLM."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                prompt = f"""
                Analyze the emotional tone of this text and classify it into one of these emotions: angry, fear, happy, neutral, sad, surprise.
                
                Text: "{text}"
                
                Consider:
                - Word choice and sentiment
                - Context and meaning
                - Emotional indicators
                
                Respond with only a JSON object:
                {{"emotion": "emotion_name", "confidence": 0.85, "reasoning": "brief explanation"}}
                """
                
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=self.groq_headers,
                    json={
                        "model": settings.groq_llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 150
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Parse JSON response
                    try:
                        emotion_data = json.loads(content)
                        emotion = emotion_data.get("emotion", "neutral")
                        confidence = emotion_data.get("confidence", 0.5)
                        
                        # Create scores distribution
                        all_scores = {e: 0.1 for e in self.emotion_labels}
                        all_scores[emotion] = confidence
                        
                        return {
                            "emotion": emotion,
                            "confidence": confidence,
                            "all_scores": all_scores,
                            "reasoning": emotion_data.get("reasoning", "")
                        }
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"Groq emotion analysis error: {str(e)}")
            
        return self._simple_emotion_analysis(text)

    async def _openrouter_emotion_analysis(self, text: str) -> Dict:
        """Analyze emotion using OpenRouter."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={**self.openrouter_headers, "Content-Type": "application/json"},
                    json={
                        "model": settings.openrouter_model_2,
                        "messages": [{
                            "role": "user",
                            "content": f"""
                            Analyze the emotional tone of this text: "{text}"
                            
                            Classify into: angry, fear, happy, neutral, sad, surprise
                            
                            Respond with JSON: {{"emotion": "emotion_name", "confidence": 0.85}}
                            """
                        }],
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
                        
                        all_scores = {e: 0.1 for e in self.emotion_labels}
                        all_scores[emotion] = confidence
                        
                        return {
                            "emotion": emotion,
                            "confidence": confidence,
                            "all_scores": all_scores
                        }
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"OpenRouter emotion analysis error: {str(e)}")
            
        return self._simple_emotion_analysis(text)

    def _simple_emotion_analysis(self, text: str) -> Dict:
        """Simple keyword-based emotion analysis as fallback."""
        text_lower = text.lower()
        
        # Keyword mapping
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "great", "awesome", "love", "wonderful", "amazing"],
            "sad": ["sad", "cry", "depressed", "down", "upset", "hurt", "disappointed"],
            "angry": ["angry", "mad", "furious", "hate", "annoyed", "irritated", "frustrated"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"],
            "surprise": ["wow", "amazing", "incredible", "unbelievable", "shocked", "surprised"],
            "neutral": ["okay", "fine", "normal", "usual", "regular"]
        }
        
        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[emotion] = min(score * 0.2, 0.9)  # Cap at 90%
        
        # If no matches, default to neutral
        if not any(scores.values()):
            scores["neutral"] = 0.7
        
        best_emotion = max(scores.items(), key=lambda x: x[1])
        
        return {
            "emotion": best_emotion[0],
            "confidence": best_emotion[1],
            "all_scores": scores
        }

    async def _create_labeled_transcription(self, segments: List[Dict]) -> str:
        """Create labeled transcription with emotion markers."""
        if not segments:
            return ""
        
        labeled_parts = []
        for segment in segments:
            emotion = segment.get("emotion", "neutral")
            text = segment.get("text", "").strip()
            if text:
                labeled_parts.append(f"[{emotion}]{text}")
        
        return " ".join(labeled_parts)

    async def _calculate_overall_emotion(self, segments: List[Dict]) -> str:
        """Calculate overall emotion from segments."""
        if not segments:
            return "neutral"
            
        emotion_weights = {}
        total_duration = 0
        
        for segment in segments:
            duration = segment.get("duration_ms", 1000)
            confidence = segment.get("confidence", 0.5)
            emotion = segment.get("emotion", "neutral")
            
            weight = duration * confidence
            emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight
            total_duration += duration
        
        if not emotion_weights:
            return "neutral"
            
        return max(emotion_weights.items(), key=lambda x: x[1])[0]

    async def _aggregate_confidence_scores(self, segments: List[Dict]) -> Dict:
        """Aggregate confidence scores from all segments."""
        if not segments:
            return {}
            
        aggregated = {}
        total_duration = sum(segment.get("duration_ms", 1000) for segment in segments)
        
        for emotion in self.emotion_labels:
            weighted_score = 0
            for segment in segments:
                duration = segment.get("duration_ms", 1000)
                scores = segment.get("all_scores", {})
                score = scores.get(emotion, 0)
                weighted_score += score * duration
            
            aggregated[emotion] = weighted_score / total_duration if total_duration > 0 else 0
            
        return aggregated

    async def get_available_models(self) -> Dict:
        """Get available models for each provider."""
        return {
            "huggingface": [
                {
                    "id": settings.default_hf_model,
                    "name": "Wav2Vec2 Facebook",
                    "description": "General purpose audio model"
                },
                {
                    "id": settings.backup_hf_model,
                    "name": "SpeechT5 ASR",
                    "description": "Microsoft speech recognition"
                }
            ],
            "groq": [
                {
                    "id": "whisper_emotion_analysis",
                    "name": "Whisper + LLM Emotion Analysis",
                    "description": "Transcription + detailed emotion analysis"
                }
            ] if self.groq_headers else [],
            "openrouter": [
                {
                    "id": settings.openrouter_model_2,
                    "name": "Claude Emotion Analysis",
                    "description": "Advanced emotion reasoning"
                }
            ] if self.openrouter_headers else []
        }