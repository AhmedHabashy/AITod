import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from backend.app.services.srt_generator import SRTGenerator, srt_generator


class TestSRTGenerator:
    """Test suite for SRTGenerator service."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_translated_segments(self):
        """Sample translated segments for testing."""
        return [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "translated_text": "Hola mundo"
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "This is a test",
                "translated_text": "Esta es una prueba"
            },
            {
                "start": 5.0,
                "end": 7.5,
                "text": "Testing SRT generation",
                "translated_text": "Probando generación de SRT"
            }
        ]

    @pytest.fixture
    def sample_segments(self):
        """Sample segments without translation for testing."""
        return [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"}
        ]

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamps
        assert srt_generator.format_timestamp(0.0) == "00:00:00,000"
        assert srt_generator.format_timestamp(2.5) == "00:00:02,500"
        assert srt_generator.format_timestamp(65.123) == "00:01:05,123"
        assert srt_generator.format_timestamp(3661.5) == "01:01:01,500"

    def test_create_subtitle(self):
        """Test subtitle creation."""
        subtitle = srt_generator.create_subtitle(
            index=1,
            start_time=0.0,
            end_time=2.5,
            text="Hello world"
        )

        assert subtitle.index == 1
        assert subtitle.start.total_seconds() == 0.0
        assert subtitle.end.total_seconds() == 2.5
        assert subtitle.content == "Hello world"

    def test_generate_srt_from_segments_with_translation(self, sample_translated_segments):
        """Test generating SRT from translated segments."""
        srt_content = srt_generator.generate_srt_from_segments(
            sample_translated_segments,
            use_translated=True
        )

        # Verify SRT format
        assert "1\n" in srt_content
        assert "00:00:00,000 --> 00:00:02,500" in srt_content
        assert "Hola mundo" in srt_content
        assert "2\n" in srt_content
        assert "Esta es una prueba" in srt_content

    def test_generate_srt_from_segments_without_translation(self, sample_segments):
        """Test generating SRT from original text."""
        srt_content = srt_generator.generate_srt_from_segments(
            sample_segments,
            use_translated=False
        )

        # Verify SRT format
        assert "1\n" in srt_content
        assert "00:00:00,000 --> 00:00:02,500" in srt_content
        assert "Hello world" in srt_content

    def test_generate_srt_empty_segments_raises_error(self):
        """Test generating SRT from empty segments raises error."""
        with pytest.raises(ValueError, match="Cannot generate SRT from empty segments"):
            srt_generator.generate_srt_from_segments([])

    def test_generate_srt_missing_fields_raises_error(self):
        """Test generating SRT with missing fields raises error."""
        invalid_segments = [{"start": 0.0, "text": "Hello"}]  # Missing 'end'

        with pytest.raises(ValueError, match="missing 'start' or 'end' field"):
            srt_generator.generate_srt_from_segments(invalid_segments, use_translated=False)

    def test_generate_srt_missing_text_field_raises_error(self):
        """Test generating SRT with missing text field raises error."""
        invalid_segments = [{"start": 0.0, "end": 2.5}]  # Missing 'text'

        with pytest.raises(ValueError, match="missing 'text' field"):
            srt_generator.generate_srt_from_segments(invalid_segments, use_translated=False)

    @pytest.mark.asyncio
    async def test_save_srt(self, temp_dir, sample_translated_segments):
        """Test saving SRT to file."""
        output_path = temp_dir / "subtitles.srt"

        result_path = await srt_generator.save_srt(
            sample_translated_segments,
            output_path,
            use_translated=True
        )

        # Verify file was created
        assert result_path.exists()
        assert result_path == output_path

        # Verify file content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Hola mundo" in content
            assert "00:00:00,000 --> 00:00:02,500" in content

    @pytest.mark.asyncio
    async def test_load_srt(self, temp_dir, sample_translated_segments):
        """Test loading SRT from file."""
        output_path = temp_dir / "subtitles.srt"

        # First save an SRT file
        await srt_generator.save_srt(sample_translated_segments, output_path, use_translated=True)

        # Then load it back
        loaded_segments = await srt_generator.load_srt(output_path)

        # Verify loaded data
        assert len(loaded_segments) == len(sample_translated_segments)
        assert loaded_segments[0]['start'] == 0.0
        assert loaded_segments[0]['end'] == 2.5
        assert loaded_segments[0]['text'] == "Hola mundo"

    @pytest.mark.asyncio
    async def test_load_srt_file_not_found(self, temp_dir):
        """Test loading SRT from non-existent file."""
        non_existent_srt = temp_dir / "non_existent.srt"

        with pytest.raises(FileNotFoundError):
            await srt_generator.load_srt(non_existent_srt)

    def test_parse_srt(self):
        """Test parsing SRT content."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hola mundo

2
00:00:02,500 --> 00:00:05,000
Esta es una prueba
"""

        segments = srt_generator.parse_srt(srt_content)

        assert len(segments) == 2
        assert segments[0]['start'] == 0.0
        assert segments[0]['end'] == 2.5
        assert segments[0]['text'] == "Hola mundo"
        assert segments[1]['start'] == 2.5
        assert segments[1]['end'] == 5.0
        assert segments[1]['text'] == "Esta es una prueba"

    @pytest.mark.asyncio
    async def test_generate_srt_from_csv_no_output_path_or_file_id(self, temp_dir):
        """Test generating SRT from CSV fails when neither output_path nor file_id is provided."""
        csv_path = temp_dir / "transcript.csv"
        csv_path.touch()

        with pytest.raises(ValueError, match="Either output_path or file_id must be provided"):
            await srt_generator.generate_srt_from_csv(csv_path)

    @pytest.mark.asyncio
    async def test_generate_srt_from_csv_file_not_found(self, temp_dir):
        """Test generating SRT from non-existent CSV file."""
        non_existent_csv = temp_dir / "non_existent.csv"
        output_path = temp_dir / "output.srt"

        with pytest.raises(FileNotFoundError):
            await srt_generator.generate_srt_from_csv(non_existent_csv, output_path=output_path)


# Run tests with: pytest tests/test_srt_generator.py -v
# Run with coverage: pytest tests/test_srt_generator.py --cov=backend.app.services.srt_generator --cov-report=html
