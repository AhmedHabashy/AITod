from typing import List, Dict, Optional
from pathlib import Path

from app.utils import settings, llm_client, file_handler


class Translator:
    """Service for translating transcript segments with context awareness."""

    @staticmethod
    async def translate_segment(
        text: str,
        source_language: str,
        target_language: str,
        context: str = "",
        provider: Optional[str] = None
    ) -> str:
        """Translate a single text segment with context.

        Args:
            text: Text to translate
            source_language: Source language code (ISO 639-1)
            target_language: Target language code (ISO 639-1)
            context: Context about the full content for better translation
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            Translated text

        Raises:
            ValueError: If languages are not supported
            Exception: If translation fails
        """
        # Validate languages
        if source_language not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Source language '{source_language}' is not supported. "
                f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}"
            )

        if target_language not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Target language '{target_language}' is not supported. "
                f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}"
            )

        # Use llm_client to translate
        translated_text = await llm_client.translate_with_context(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
            provider=provider
        )

        return translated_text

    @staticmethod
    async def translate_segments(
        segments: List[Dict[str, any]],
        source_language: str,
        target_language: str,
        context: str = "",
        provider: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Translate all segments in a transcript with context awareness.

        Args:
            segments: List of transcript segments with 'start', 'end', 'text'
            source_language: Source language code (ISO 639-1)
            target_language: Target language code (ISO 639-1)
            context: Context about the full content for better translation
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            List of translated segments with same structure plus 'translated_text' field

        Raises:
            ValueError: If segments is empty or languages are not supported
            Exception: If translation fails
        """
        if not segments:
            raise ValueError("Cannot translate empty segments")

        translated_segments = []

        for segment in segments:
            # Translate the text
            translated_text = await Translator.translate_segment(
                text=segment['text'],
                source_language=source_language,
                target_language=target_language,
                context=context,
                provider=provider
            )

            # Create new segment with translation
            translated_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'translated_text': translated_text
            }

            translated_segments.append(translated_segment)

        return translated_segments

    @staticmethod
    async def translate_segments_batch(
        segments: List[Dict[str, any]],
        source_language: str,
        target_language: str,
        context: str = "",
        batch_size: int = 5,
        provider: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Translate segments in batches for better performance.

        This method processes segments in batches to optimize API calls
        while maintaining context awareness.

        Args:
            segments: List of transcript segments
            source_language: Source language code (ISO 639-1)
            target_language: Target language code (ISO 639-1)
            context: Context about the full content
            batch_size: Number of segments to process together (not used yet, reserved for future optimization)
            provider: LLM provider to use

        Returns:
            List of translated segments

        Note:
            Currently processes segments one by one. Batch processing can be
            implemented in the future to optimize API calls.
        """
        # For now, this is the same as translate_segments
        # In the future, we could optimize by combining multiple segments
        # into a single API call
        return await Translator.translate_segments(
            segments=segments,
            source_language=source_language,
            target_language=target_language,
            context=context,
            provider=provider
        )

    @staticmethod
    async def translate_from_csv(
        csv_path: Path,
        source_language: str,
        target_language: str,
        context: str = "",
        provider: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Translate segments from a CSV transcript file.

        Args:
            csv_path: Path to the CSV transcript file
            source_language: Source language code (ISO 639-1)
            target_language: Target language code (ISO 639-1)
            context: Context about the full content
            provider: LLM provider to use

        Returns:
            List of translated segments

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            Exception: If translation fails
        """
        # Import here to avoid circular dependency
        from .transcriber import transcriber

        # Load transcript from CSV
        segments = await transcriber.load_transcript_from_csv(csv_path)

        # Translate segments
        translated_segments = await Translator.translate_segments(
            segments=segments,
            source_language=source_language,
            target_language=target_language,
            context=context,
            provider=provider
        )

        return translated_segments

    @staticmethod
    async def save_translated_segments_to_csv(
        translated_segments: List[Dict[str, any]],
        output_path: Path
    ) -> Path:
        """Save translated segments to CSV file.

        Args:
            translated_segments: List of segments with 'translated_text' field
            output_path: Path where CSV file will be saved

        Returns:
            Path to the saved CSV file

        Raises:
            ValueError: If translated_segments is empty
            Exception: If saving fails
        """
        if not translated_segments:
            raise ValueError("Cannot save empty translated segments")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import csv

            # Write to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['start_time', 'end_time', 'original_text', 'translated_text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write segments
                for segment in translated_segments:
                    writer.writerow({
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'original_text': segment['text'],
                        'translated_text': segment['translated_text']
                    })

            return output_path

        except Exception as e:
            raise Exception(f"Failed to save translated segments to CSV: {str(e)}")


# Create singleton instance
translator = Translator()
