import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from backend.app.services.audio_extractor import AudioExtractor, audio_extractor


class TestAudioExtractor:
    """Test suite for AudioExtractor service."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_check_ffmpeg_installed(self):
        """Test that FFmpeg is installed and accessible."""
        assert AudioExtractor.check_ffmpeg_installed(), "FFmpeg is not installed or not in PATH"

    @pytest.mark.asyncio
    async def test_extract_audio_file_not_found(self, temp_dir):
        """Test extraction fails when video file doesn't exist."""
        non_existent_video = temp_dir / "non_existent.mp4"

        with pytest.raises(FileNotFoundError):
            await audio_extractor.extract_audio(non_existent_video)

    @pytest.mark.asyncio
    async def test_extract_audio_no_output_path_or_file_id(self, temp_dir):
        """Test extraction fails when neither output_path nor file_id is provided."""
        # Create a dummy video file (we won't actually extract from it)
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        with pytest.raises(ValueError, match="Either output_path or file_id must be provided"):
            await audio_extractor.extract_audio(video_path)

    @pytest.mark.asyncio
    async def test_get_audio_duration_file_not_found(self, temp_dir):
        """Test getting duration fails when audio file doesn't exist."""
        non_existent_audio = temp_dir / "non_existent.wav"

        with pytest.raises(FileNotFoundError):
            await audio_extractor.get_audio_duration(non_existent_audio)


# Run tests with: pytest tests/test_audio_extractor.py -v
# Run with coverage: pytest tests/test_audio_extractor.py --cov=backend.app.services.audio_extractor --cov-report=html
