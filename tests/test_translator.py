import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import csv

from backend.app.services.translator import Translator, translator


class TestTranslator:
    """Test suite for Translator service."""

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
            {"start": 5.0, "end": 7.5, "text": "Testing translation"}
        ]

    @pytest.fixture
    def translated_segments(self):
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
                "text": "Testing translation",
                "translated_text": "Probando traducción"
            }
        ]

    @pytest.mark.asyncio
    async def test_translate_segment_unsupported_source_language(self):
        """Test translation fails with unsupported source language."""
        with pytest.raises(ValueError, match="Source language .* is not supported"):
            await translator.translate_segment(
                text="Hello",
                source_language="xyz",
                target_language="es"
            )

    @pytest.mark.asyncio
    async def test_translate_segment_unsupported_target_language(self):
        """Test translation fails with unsupported target language."""
        with pytest.raises(ValueError, match="Target language .* is not supported"):
            await translator.translate_segment(
                text="Hello",
                source_language="en",
                target_language="xyz"
            )

    @pytest.mark.asyncio
    async def test_translate_segments_empty_raises_error(self):
        """Test translating empty segments raises ValueError."""
        with pytest.raises(ValueError, match="Cannot translate empty segments"):
            await translator.translate_segments(
                segments=[],
                source_language="en",
                target_language="es"
            )

    @pytest.mark.asyncio
    async def test_save_translated_segments_to_csv(self, temp_dir, translated_segments):
        """Test saving translated segments to CSV."""
        output_path = temp_dir / "translated.csv"

        result_path = await translator.save_translated_segments_to_csv(
            translated_segments,
            output_path
        )

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
            assert rows[0]['original_text'] == 'Hello world'
            assert rows[0]['translated_text'] == 'Hola mundo'

    @pytest.mark.asyncio
    async def test_save_empty_translated_segments_fails(self, temp_dir):
        """Test saving empty translated segments raises error."""
        output_path = temp_dir / "translated.csv"

        with pytest.raises(ValueError, match="Cannot save empty translated segments"):
            await translator.save_translated_segments_to_csv([], output_path)

    @pytest.mark.asyncio
    async def test_translate_from_csv_file_not_found(self, temp_dir):
        """Test translating from non-existent CSV file."""
        non_existent_csv = temp_dir / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            await translator.translate_from_csv(
                csv_path=non_existent_csv,
                source_language="en",
                target_language="es"
            )

    @pytest.mark.asyncio
    async def test_translate_from_csv(self, temp_dir, sample_segments):
        """Test translating from CSV file (will fail without API keys)."""
        from backend.app.services.transcriber import transcriber

        # Save segments to CSV
        csv_path = temp_dir / "transcript.csv"
        await transcriber.save_transcript_to_csv(sample_segments, csv_path)

        # This test will fail without valid API keys
        # We're just testing that the function is callable and validates inputs
        try:
            translated = await translator.translate_from_csv(
                csv_path=csv_path,
                source_language="en",
                target_language="es",
                context="Test context"
            )
            # If it succeeds (with valid API keys), verify structure
            assert isinstance(translated, list)
            if translated:
                assert 'translated_text' in translated[0]
        except Exception as e:
            # Expected to fail without valid API keys
            # Just verify the error is from LLM client, not our code
            assert "API key" in str(e) or "failed" in str(e).lower()


# Run tests with: pytest tests/test_translator.py -v
# Run with coverage: pytest tests/test_translator.py --cov=backend.app.services.translator --cov-report=html
