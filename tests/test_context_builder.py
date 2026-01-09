import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from backend.app.services.context_builder import ContextBuilder, context_builder


class TestContextBuilder:
    """Test suite for ContextBuilder service."""

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
            {"start": 0.0, "end": 3.0, "text": "Welcome to this Python programming tutorial"},
            {"start": 3.0, "end": 6.0, "text": "Today we'll learn about machine learning"},
            {"start": 6.0, "end": 9.0, "text": "We'll use TensorFlow and PyTorch libraries"},
            {"start": 9.0, "end": 12.0, "text": "Let's start with neural networks"}
        ]

    @pytest.fixture
    def short_segments(self):
        """Short transcript segments for testing edge cases."""
        return [
            {"start": 0.0, "end": 1.0, "text": "Hi"}
        ]

    def test_build_simple_context(self, sample_segments):
        """Test building simple context without LLM."""
        context = context_builder.build_simple_context(sample_segments, source_language="en")

        assert "en" in context
        assert "12.0 seconds" in context
        assert "words" in context

    def test_build_simple_context_empty_segments(self):
        """Test building simple context with empty segments."""
        context = context_builder.build_simple_context([], source_language="en")

        assert context == "Empty transcript."

    def test_build_simple_context_calculates_duration(self, sample_segments):
        """Test that simple context correctly calculates duration."""
        context = context_builder.build_simple_context(sample_segments)

        # Duration should be from first segment start to last segment end
        # 0.0 to 12.0 = 12.0 seconds
        assert "12.0 seconds" in context

    def test_build_simple_context_counts_words(self, sample_segments):
        """Test that simple context counts words approximately."""
        context = context_builder.build_simple_context(sample_segments)

        # Should mention word count
        assert "word" in context.lower()

    @pytest.mark.asyncio
    async def test_build_context_empty_segments_raises_error(self):
        """Test that building context with empty segments raises ValueError."""
        with pytest.raises(ValueError, match="Cannot build context from empty segments"):
            await context_builder.build_context([])

    @pytest.mark.asyncio
    async def test_build_context_short_text(self, short_segments):
        """Test building context with very short text."""
        # Short text should return a simple default context
        context = await context_builder.build_context(short_segments)

        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_build_context_from_file_not_found(self, temp_dir):
        """Test building context from non-existent CSV file."""
        non_existent_csv = temp_dir / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            await context_builder.build_context_from_file(non_existent_csv)

    @pytest.mark.asyncio
    async def test_build_context_from_file(self, temp_dir, sample_segments):
        """Test building context from CSV file."""
        from backend.app.services.transcriber import transcriber

        # Save segments to CSV
        csv_path = temp_dir / "transcript.csv"
        await transcriber.save_transcript_to_csv(sample_segments, csv_path)

        # Build context from file
        context = await context_builder.build_context_from_file(csv_path, source_language="en")

        assert isinstance(context, str)
        assert len(context) > 0


# Run tests with: pytest tests/test_context_builder.py -v
# Run with coverage: pytest tests/test_context_builder.py --cov=backend.app.services.context_builder --cov-report=html
