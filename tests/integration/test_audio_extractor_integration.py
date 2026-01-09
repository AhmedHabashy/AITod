"""
Integration tests for AudioExtractor service.

These tests verify actual audio extraction functionality with real video files.
Requires FFmpeg to be installed.

Run with: pytest tests/integration/test_audio_extractor_integration.py -v -s
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import numpy as np
from pydub import AudioSegment

from backend.app.services.audio_extractor import audio_extractor


class TestAudioExtractorIntegration:
    """Integration tests for audio extraction functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_video_with_audio(self, temp_dir):
        """Create a sample video file with audio for testing."""
        video_path = temp_dir / "sample_video.mp4"

        try:
            import moviepy.editor as mp

            # Create a 5-second black video with a tone
            def make_frame(t):
                return np.zeros((480, 640, 3), dtype=np.uint8)

            # Create video clip
            video = mp.VideoClip(make_frame, duration=5)
            video.fps = 24

            # Create audio (sine wave)
            audio_freq = 440  # A4 note
            sample_rate = 44100
            duration = 5
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * audio_freq * t)

            # Create audio clip
            from moviepy.audio.AudioClip import AudioClip
            def make_audio(t):
                idx = int(t * sample_rate)
                if idx < len(audio_data):
                    return audio_data[idx]
                return 0

            audio = AudioClip(make_audio, duration=duration, fps=sample_rate)
            video_with_audio = video.set_audio(audio)

            # Write video file
            video_with_audio.write_videofile(
                str(video_path),
                codec="libx264",
                audio_codec="aac",
                fps=24,
                verbose=False,
                logger=None
            )

            # Cleanup
            video.close()
            audio.close()
            video_with_audio.close()

            return video_path

        except Exception as e:
            pytest.skip(f"Could not create sample video: {str(e)}")

    @pytest.mark.asyncio
    async def test_extract_audio_from_video(self, temp_dir, sample_video_with_audio):
        """Test extracting audio from a real video file."""
        print(f"\n\nTest: Extracting audio from video")
        print(f"Video path: {sample_video_with_audio}")

        if not sample_video_with_audio.exists():
            pytest.skip("Sample video not available")

        output_path = temp_dir / "extracted_audio.wav"

        # Extract audio
        result_path = await audio_extractor.extract_audio(
            sample_video_with_audio,
            output_path
        )

        print(f"Audio extracted to: {result_path}")

        # Verify audio file was created
        assert result_path.exists(), "Audio file was not created"
        assert result_path.suffix == ".wav", "Audio file is not WAV format"
        assert result_path.stat().st_size > 0, "Audio file is empty"

        # Verify audio properties
        audio = AudioSegment.from_wav(str(result_path))
        print(f"Audio duration: {audio.duration_seconds:.2f}s")
        print(f"Sample rate: {audio.frame_rate}Hz")
        print(f"Channels: {audio.channels}")

        assert audio.duration_seconds > 4, "Audio duration is too short"
        assert audio.duration_seconds < 6, "Audio duration is too long"
        assert audio.frame_rate == 16000, "Audio sample rate is not 16kHz"
        assert audio.channels == 1, "Audio is not mono"

        print("✓ Audio extraction successful\n")

    @pytest.mark.asyncio
    async def test_extract_audio_with_file_id(self, temp_dir, sample_video_with_audio):
        """Test extracting audio using file_id parameter."""
        print(f"\n\nTest: Extract audio with file_id")

        if not sample_video_with_audio.exists():
            pytest.skip("Sample video not available")

        file_id = "test_video_123"

        # Extract audio using file_id
        result_path = await audio_extractor.extract_audio(
            sample_video_with_audio,
            file_id=file_id
        )

        print(f"Audio extracted to: {result_path}")

        # Verify file was created in correct location
        assert result_path.exists(), "Audio file was not created"
        assert file_id in str(result_path), "File ID not in path"

        print("✓ Audio extraction with file_id successful\n")

    @pytest.mark.asyncio
    async def test_get_audio_duration_from_extracted(self, temp_dir, sample_video_with_audio):
        """Test getting duration from extracted audio file."""
        print(f"\n\nTest: Get audio duration")

        if not sample_video_with_audio.exists():
            pytest.skip("Sample video not available")

        # First extract audio
        audio_path = temp_dir / "audio.wav"
        await audio_extractor.extract_audio(sample_video_with_audio, audio_path)

        # Get duration
        duration = await audio_extractor.get_audio_duration(audio_path)

        print(f"Audio duration: {duration:.2f}s")

        # Should be approximately 5 seconds
        assert 4.5 <= duration <= 5.5, f"Duration {duration}s is not close to expected 5s"

        print("✓ Duration extraction successful\n")

    @pytest.mark.asyncio
    async def test_extract_audio_formats(self, temp_dir):
        """Test audio extraction from different video formats."""
        print(f"\n\nTest: Extract audio from different formats")

        # This test would require different video formats
        # For now, we'll just verify the method exists and handles formats
        # You can add actual format tests if you have sample videos

        pytest.skip("Requires multiple video format samples")


# Run tests with:
# pytest tests/integration/test_audio_extractor_integration.py -v -s
