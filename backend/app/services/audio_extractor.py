import subprocess
from pathlib import Path
from typing import Optional

from app.utils import settings, file_handler


class AudioExtractor:
    """Service for extracting audio from video files using FFmpeg."""

    @staticmethod
    async def extract_audio(
        video_path: Path,
        output_path: Optional[Path] = None,
        file_id: Optional[str] = None
    ) -> Path:
        """Extract audio from video file and save as WAV.

        Args:
            video_path: Path to the input video file
            output_path: Optional path for output audio file
            file_id: Optional file ID, required if output_path is not provided

        Returns:
            Path to the extracted audio file

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If FFmpeg extraction fails
            ValueError: If neither output_path nor file_id is provided
        """
        # Validate input file exists
        if not file_handler.file_exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Determine output path
        if output_path is None:
            if file_id is None:
                raise ValueError("Either output_path or file_id must be provided")
            output_path = file_handler.get_file_path(file_id, "audio", "wav")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # FFmpeg command to extract audio
        # -i: input file
        # -vn: disable video
        # -acodec pcm_s16le: use PCM 16-bit little-endian codec (high quality WAV)
        # -ar 16000: set audio sample rate to 16kHz (good for speech recognition)
        # -ac 1: convert to mono (reduces file size, good for speech)
        # -y: overwrite output file if exists
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output_path)
        ]

        try:
            # Run FFmpeg command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            # Verify output file was created
            if not file_handler.file_exists(output_path):
                raise RuntimeError("FFmpeg completed but output file was not created")

            return output_path

        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg failed to extract audio: {e.stderr}"
            raise RuntimeError(error_message)

        except Exception as e:
            raise RuntimeError(f"Unexpected error during audio extraction: {str(e)}")

    @staticmethod
    async def extract_audio_with_timestamps(
        video_path: Path,
        output_path: Optional[Path] = None,
        file_id: Optional[str] = None
    ) -> tuple[Path, float]:
        """Extract audio from video and return audio path with duration.

        Args:
            video_path: Path to the input video file
            output_path: Optional path for output audio file
            file_id: Optional file ID, required if output_path is not provided

        Returns:
            Tuple of (audio_path, duration_in_seconds)

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If FFmpeg extraction fails
        """
        # Extract audio
        audio_path = await AudioExtractor.extract_audio(video_path, output_path, file_id)

        # Get audio duration using FFprobe
        duration = await AudioExtractor.get_audio_duration(audio_path)

        return audio_path, duration

    @staticmethod
    async def get_audio_duration(audio_path: Path) -> float:
        """Get duration of audio file in seconds using FFprobe.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If FFprobe fails
        """
        if not file_handler.file_exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # FFprobe command to get duration
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            duration = float(result.stdout.strip())
            return duration

        except subprocess.CalledProcessError as e:
            error_message = f"FFprobe failed to get audio duration: {e.stderr}"
            raise RuntimeError(error_message)

        except ValueError:
            raise RuntimeError(f"Failed to parse duration from FFprobe output: {result.stdout}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error getting audio duration: {str(e)}")

    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if FFmpeg is installed and accessible.

        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


# Create singleton instance
audio_extractor = AudioExtractor()
