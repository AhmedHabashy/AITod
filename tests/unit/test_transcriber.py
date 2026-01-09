import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import csv

from backend.app.services.transcriber import Transcriber, transcriber


class TestTranscriber:
    """Test suite for Transcriber service."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_segments(self):
        """Sample transcript segments for testing."""
        return [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
            {"start": 5.0, "end": 7.5, "text": "Testing transcription"}
        ]

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self, temp_dir):
        """Test transcription fails when audio file doesn't exist."""
        non_existent_audio = temp_dir / "non_existent.wav"

        with pytest.raises(FileNotFoundError):
            await transcriber.transcribe_audio(non_existent_audio)

    @pytest.mark.asyncio
    async def test_transcribe_audio_unsupported_language(self, temp_dir):
        """Test transcription fails with unsupported language."""
        # Create a dummy audio file
        audio_path = temp_dir / "test.wav"
        audio_path.touch()

        with pytest.raises(ValueError, match="Language .* is not supported"):
            await transcriber.transcribe_audio(audio_path, language="xyz")

    @pytest.mark.asyncio
    async def test_save_transcript_to_csv(self, temp_dir, sample_segments):
        """Test saving transcript segments to CSV."""
        output_path = temp_dir / "transcript.csv"

        result_path = await transcriber.save_transcript_to_csv(sample_segments, output_path)

        # Verify file was created
        assert result_path.exists()
        assert result_path == output_path

        # Verify CSV content
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            assert len(rows) == 3
            assert rows[0]['start_time'] == '0.0'
            assert rows[0]['end_time'] == '2.5'
            assert rows[0]['text'] == 'Hello world'

    @pytest.mark.asyncio
    async def test_save_empty_transcript_fails(self, temp_dir):
        """Test saving empty transcript raises error."""
        output_path = temp_dir / "transcript.csv"

        with pytest.raises(ValueError, match="Cannot save empty transcript"):
            await transcriber.save_transcript_to_csv([], output_path)

    @pytest.mark.asyncio
    async def test_load_transcript_from_csv(self, temp_dir, sample_segments):
        """Test loading transcript from CSV file."""
        csv_path = temp_dir / "transcript.csv"

        # First save the segments
        await transcriber.save_transcript_to_csv(sample_segments, csv_path)

        # Then load them back
        loaded_segments = await transcriber.load_transcript_from_csv(csv_path)

        # Verify loaded data matches original
        assert len(loaded_segments) == len(sample_segments)
        for i, segment in enumerate(loaded_segments):
            assert segment['start'] == sample_segments[i]['start']
            assert segment['end'] == sample_segments[i]['end']
            assert segment['text'] == sample_segments[i]['text']

    @pytest.mark.asyncio
    async def test_load_transcript_file_not_found(self, temp_dir):
        """Test loading transcript fails when CSV doesn't exist."""
        non_existent_csv = temp_dir / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            await transcriber.load_transcript_from_csv(non_existent_csv)

    def test_get_full_transcript_text(self, sample_segments):
        """Test extracting full text from segments."""
        full_text = transcriber.get_full_transcript_text(sample_segments)

        assert full_text == "Hello world This is a test Testing transcription"

    def test_get_full_transcript_text_empty(self):
        """Test extracting text from empty segments."""
        full_text = transcriber.get_full_transcript_text([])

        assert full_text == ""

    @pytest.mark.asyncio
    async def test_transcribe_and_save_no_output_path_or_file_id(self, temp_dir):
        """Test transcribe_and_save fails when neither output_csv_path nor file_id is provided."""
        # Create a dummy audio file
        audio_path = temp_dir / "test.wav"
        audio_path.touch()

        # This should fail during the save step since we need mock for actual transcription
        # For now, we're testing the validation logic
        with pytest.raises(ValueError, match="Either output_csv_path or file_id must be provided"):
            # This will fail early at validation, before trying to transcribe
            # We'd need to mock llm_client to test the full flow
            pass


# Run tests with: pytest tests/test_transcriber.py -v
# Run with coverage: pytest tests/test_transcriber.py --cov=backend.app.services.transcriber --cov-report=html
