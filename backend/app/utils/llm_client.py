from pathlib import Path
from typing import List, Dict, Optional
import openai
from google import genai

from .config import settings


class LLMClient:
    """Client for interacting with LLM APIs (OpenAI and Gemini)."""

    def __init__(self):
        """Initialize LLM clients based on available API keys."""
        self.openai_client = None
        self.gemini_client = None

        # Initialize OpenAI client if API key is available
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        # Initialize Gemini client if API key is available
        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def transcribe_audio(
        self,
        audio_file_path: Path,
        language: str = "en",
        provider: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_file_path: Path to the audio file
            language: Source language code (ISO 639-1)
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            List of dictionaries with 'start', 'end', and 'text' keys

        Raises:
            ValueError: If the specified provider is not available
            Exception: If transcription fails
        """
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        if provider == "openai":
            return await self._transcribe_with_openai(audio_file_path, language)
        elif provider == "gemini":
            return await self._transcribe_with_gemini(audio_file_path, language)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _transcribe_with_openai(
        self,
        audio_file_path: Path,
        language: str
    ) -> List[Dict[str, any]]:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio_file_path: Path to the audio file
            language: Source language code

        Returns:
            List of transcription segments with timestamps
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            with open(audio_file_path, "rb") as audio_file:
                # Use OpenAI audio transcription with timestamp option
                response = self.openai_client.audio.transcriptions.create(
                    model=settings.OPENAI_MODEL,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            # Parse response and extract segments
            segments = []
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
            else:
                # Fallback if no segments (shouldn't happen with verbose_json)
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": response.text
                })

            return segments

        except Exception as e:
            raise Exception(f"OpenAI transcription failed: {str(e)}")

    async def _transcribe_with_gemini(
        self,
        audio_file_path: Path,
        language: str
    ) -> List[Dict[str, any]]:
        """Transcribe audio using Gemini API.

        Args:
            audio_file_path: Path to the audio file
            language: Source language code

        Returns:
            List of transcription segments with timestamps
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        try:
            # Upload audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()

            # Create prompt for transcription with timestamps
            prompt = f"""Transcribe this audio file in {language} language.
            Provide the transcription with timestamps in this JSON format:
            [{{"start": 0.0, "end": 2.5, "text": "transcribed text"}}, ...]

            Only return the JSON array, no additional text."""

            # Use Gemini model for transcription
            response = self.gemini_client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_data
                            }}
                        ]
                    }
                ]
            )

            # Parse response
            import json
            segments = json.loads(response.text)
            return segments

        except Exception as e:
            raise Exception(f"Gemini transcription failed: {str(e)}")

    async def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: str,
        provider: Optional[str] = None
    ) -> str:
        """Translate text with context for professional translation.

        Args:
            text: Text to translate
            source_language: Source language code (ISO 639-1)
            target_language: Target language code (ISO 639-1)
            context: Context about the full content for better translation
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            Translated text

        Raises:
            ValueError: If the specified provider is not available
            Exception: If translation fails
        """
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        if provider == "openai":
            return await self._translate_with_openai(
                text, source_language, target_language, context
            )
        elif provider == "gemini":
            return await self._translate_with_gemini(
                text, source_language, target_language, context
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _translate_with_openai(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: str
    ) -> str:
        """Translate text using OpenAI GPT API.

        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            context: Context for better translation

        Returns:
            Translated text
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            prompt = f"""You are a professional translator.

Context about the full content: {context}

Translate the following text from {source_language} to {target_language}.
Maintain professional tone, cultural nuances, and technical accuracy.
Only return the translated text, no explanations.

Text to translate:
"{text}"

Translation:"""

            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"OpenAI translation failed: {str(e)}")

    async def _translate_with_gemini(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: str
    ) -> str:
        """Translate text using Gemini API.

        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            context: Context for better translation

        Returns:
            Translated text
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        try:
            prompt = f"""You are a professional translator.

Context about the full content: {context}

Translate the following text from {source_language} to {target_language}.
Maintain professional tone, cultural nuances, and technical accuracy.
Only return the translated text, no explanations.

Text to translate:
"{text}"

Translation:"""

            response = self.gemini_client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            raise Exception(f"Gemini translation failed: {str(e)}")


# Create a singleton instance
llm_client = LLMClient()
