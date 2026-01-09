from typing import List, Dict, Optional
from pathlib import Path
import srt
from datetime import timedelta

from app.utils import file_handler


class SRTGenerator:
    """Service for generating SRT subtitle files from translated segments."""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string (e.g., "00:00:05,500")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def create_subtitle(
        index: int,
        start_time: float,
        end_time: float,
        text: str
    ) -> srt.Subtitle:
        """Create an SRT subtitle object.

        Args:
            index: Subtitle number (1-indexed)
            start_time: Start time in seconds
            end_time: End time in seconds
            text: Subtitle text

        Returns:
            srt.Subtitle object
        """
        return srt.Subtitle(
            index=index,
            start=timedelta(seconds=start_time),
            end=timedelta(seconds=end_time),
            content=text
        )

    @staticmethod
    def generate_srt_from_segments(
        segments: List[Dict[str, any]],
        use_translated: bool = True
    ) -> str:
        """Generate SRT content from segments.

        Args:
            segments: List of segments with timestamps and text
            use_translated: If True, use 'translated_text' field, else use 'text'

        Returns:
            SRT formatted string

        Raises:
            ValueError: If segments is empty or missing required fields
        """
        if not segments:
            raise ValueError("Cannot generate SRT from empty segments")

        # Determine which text field to use
        text_field = 'translated_text' if use_translated else 'text'

        # Validate segments have required fields
        for i, segment in enumerate(segments):
            if 'start' not in segment or 'end' not in segment:
                raise ValueError(f"Segment {i} missing 'start' or 'end' field")
            if text_field not in segment:
                raise ValueError(f"Segment {i} missing '{text_field}' field")

        # Create subtitle objects
        subtitles = []
        for i, segment in enumerate(segments, start=1):
            subtitle = SRTGenerator.create_subtitle(
                index=i,
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment[text_field]
            )
            subtitles.append(subtitle)

        # Generate SRT content
        srt_content = srt.compose(subtitles)
        return srt_content

    @staticmethod
    async def save_srt(
        segments: List[Dict[str, any]],
        output_path: Path,
        use_translated: bool = True
    ) -> Path:
        """Generate and save SRT file from segments.

        Args:
            segments: List of segments with timestamps and text
            output_path: Path where SRT file will be saved
            use_translated: If True, use 'translated_text' field, else use 'text'

        Returns:
            Path to the saved SRT file

        Raises:
            ValueError: If segments is empty or invalid
            Exception: If saving fails
        """
        # Generate SRT content
        srt_content = SRTGenerator.generate_srt_from_segments(segments, use_translated)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            return output_path

        except Exception as e:
            raise Exception(f"Failed to save SRT file: {str(e)}")

    @staticmethod
    async def generate_srt_from_csv(
        csv_path: Path,
        output_path: Optional[Path] = None,
        file_id: Optional[str] = None,
        use_translated: bool = True
    ) -> Path:
        """Generate SRT file from CSV transcript/translation file.

        Args:
            csv_path: Path to the CSV file with segments
            output_path: Optional path for output SRT file
            file_id: Optional file ID, required if output_path is not provided
            use_translated: If True, expect translated segments, else use original text

        Returns:
            Path to the saved SRT file

        Raises:
            ValueError: If neither output_path nor file_id is provided
            FileNotFoundError: If CSV file doesn't exist
            Exception: If generation fails
        """
        # Import here to avoid circular dependency
        if use_translated:
            # Load translated segments from CSV
            import csv
            if not file_handler.file_exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            segments = []
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    segments.append({
                        'start': float(row['start_time']),
                        'end': float(row['end_time']),
                        'text': row.get('original_text', ''),
                        'translated_text': row['translated_text']
                    })
        else:
            # Load original transcript
            from .transcriber import transcriber
            segments = await transcriber.load_transcript_from_csv(csv_path)

        # Determine output path
        if output_path is None:
            if file_id is None:
                raise ValueError("Either output_path or file_id must be provided")
            output_path = file_handler.get_file_path(file_id, "output", "srt")

        # Save SRT
        srt_path = await SRTGenerator.save_srt(segments, output_path, use_translated)

        return srt_path

    @staticmethod
    def parse_srt(srt_content: str) -> List[Dict[str, any]]:
        """Parse SRT content into segments.

        Args:
            srt_content: SRT formatted string

        Returns:
            List of dictionaries with 'start', 'end', 'text' keys

        Raises:
            Exception: If parsing fails
        """
        try:
            subtitles = srt.parse(srt_content)

            segments = []
            for subtitle in subtitles:
                segments.append({
                    'start': subtitle.start.total_seconds(),
                    'end': subtitle.end.total_seconds(),
                    'text': subtitle.content
                })

            return segments

        except Exception as e:
            raise Exception(f"Failed to parse SRT content: {str(e)}")

    @staticmethod
    async def load_srt(srt_path: Path) -> List[Dict[str, any]]:
        """Load and parse SRT file.

        Args:
            srt_path: Path to the SRT file

        Returns:
            List of dictionaries with 'start', 'end', 'text' keys

        Raises:
            FileNotFoundError: If SRT file doesn't exist
            Exception: If loading or parsing fails
        """
        if not file_handler.file_exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            segments = SRTGenerator.parse_srt(srt_content)
            return segments

        except Exception as e:
            raise Exception(f"Failed to load SRT file: {str(e)}")


# Create singleton instance
srt_generator = SRTGenerator()
