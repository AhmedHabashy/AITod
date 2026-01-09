import csv
from pathlib import Path
from typing import List, Dict, Optional

from app.utils import settings, file_handler, llm_client


class Transcriber:
    """Service for transcribing audio files to text with timestamps."""

    @staticmethod
    async def transcribe_audio(
        audio_path: Path,
        language: str = "en",
        provider: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to the audio file
            language: Source language code (ISO 639-1)
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            List of dictionaries with 'start', 'end', and 'text' keys

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If language is not supported or provider is invalid
            Exception: If transcription fails
        """
        # Validate audio file exists
        if not file_handler.file_exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate language is supported
        if language not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}"
            )

        # Use llm_client to transcribe
        segments = await llm_client.transcribe_audio(
            audio_file_path=audio_path,
            language=language,
            provider=provider
        )

        return segments

    @staticmethod
    async def transcribe_and_save(
        audio_path: Path,
        output_csv_path: Optional[Path] = None,
        file_id: Optional[str] = None,
        language: str = "en",
        provider: Optional[str] = None
    ) -> tuple[List[Dict[str, any]], Path]:
        """Transcribe audio and save to CSV file.

        Args:
            audio_path: Path to the audio file
            output_csv_path: Optional path for output CSV file
            file_id: Optional file ID, required if output_csv_path is not provided
            language: Source language code (ISO 639-1)
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            Tuple of (segments, csv_path)

        Raises:
            ValueError: If neither output_csv_path nor file_id is provided
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription or saving fails
        """
        # Transcribe audio
        segments = await Transcriber.transcribe_audio(
            audio_path=audio_path,
            language=language,
            provider=provider
        )

        # Determine output path
        if output_csv_path is None:
            if file_id is None:
                raise ValueError("Either output_csv_path or file_id must be provided")
            output_csv_path = file_handler.get_file_path(file_id, "transcript", "csv")

        # Save to CSV
        csv_path = await Transcriber.save_transcript_to_csv(segments, output_csv_path)

        return segments, csv_path

    @staticmethod
    async def save_transcript_to_csv(
        segments: List[Dict[str, any]],
        output_path: Path
    ) -> Path:
        """Save transcript segments to CSV file.

        Args:
            segments: List of transcript segments with 'start', 'end', 'text'
            output_path: Path where CSV file will be saved

        Returns:
            Path to the saved CSV file

        Raises:
            ValueError: If segments is empty or invalid
            Exception: If saving fails
        """
        if not segments:
            raise ValueError("Cannot save empty transcript")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['start_time', 'end_time', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write segments
                for segment in segments:
                    writer.writerow({
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'text': segment['text']
                    })

            return output_path

        except Exception as e:
            raise Exception(f"Failed to save transcript to CSV: {str(e)}")

    @staticmethod
    async def load_transcript_from_csv(csv_path: Path) -> List[Dict[str, any]]:
        """Load transcript segments from CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            List of dictionaries with 'start', 'end', and 'text' keys

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            Exception: If loading fails
        """
        if not file_handler.file_exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            segments = []
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    segments.append({
                        'start': float(row['start_time']),
                        'end': float(row['end_time']),
                        'text': row['text']
                    })

            return segments

        except Exception as e:
            raise Exception(f"Failed to load transcript from CSV: {str(e)}")

    @staticmethod
    def get_full_transcript_text(segments: List[Dict[str, any]]) -> str:
        """Extract full text from transcript segments.

        Args:
            segments: List of transcript segments

        Returns:
            Full transcript text as a single string
        """
        return " ".join(segment['text'] for segment in segments)


# Create singleton instance
transcriber = Transcriber()
