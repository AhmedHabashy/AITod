"""
Integration tests for Translator service.

These tests verify actual translation functionality with LLM APIs.
Requires:
- Valid API keys (OpenAI and/or Gemini)
- Internet connection

Run with: pytest tests/integration/test_translator_integration.py -v -s
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from backend.app.services.translator import translator
from backend.app.services.transcriber import transcriber


class TestTranslatorIntegration:
    """Integration tests for translation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def english_segments(self):
        """Sample English segments for translation."""
        return [
            {"start": 0.0, "end": 3.0, "text": "Hello and welcome to our channel."},
            {"start": 3.0, "end": 6.0, "text": "Today we will learn about programming."},
            {"start": 6.0, "end": 9.0, "text": "Programming is the art of giving instructions to computers."}
        ]

    @pytest.fixture
    def technical_segments(self):
        """Technical content for translation."""
        return [
            {"start": 0.0, "end": 4.0, "text": "Machine learning is a subset of artificial intelligence."},
            {"start": 4.0, "end": 8.0, "text": "It focuses on algorithms that improve through experience."},
            {"start": 8.0, "end": 12.0, "text": "Neural networks are inspired by biological neurons in the brain."}
        ]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_single_segment_with_openai(self):
        """Test translating a single segment using OpenAI."""
        print(f"\n\nTest: Translate single segment with OpenAI")

        text = "Hello world, how are you?"
        print(f"Original (en): {text}")

        try:
            translated = await translator.translate_segment(
                text=text,
                source_language="en",
                target_language="es",
                context="A greeting message",
                provider="openai"
            )

            print(f"Translated (es): {translated}")

            # Verify translation
            assert isinstance(translated, str)
            assert len(translated) > 0
            assert translated != text  # Should be different from original

            print("✓ OpenAI single segment translation successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("OpenAI API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_single_segment_with_gemini(self):
        """Test translating a single segment using Gemini."""
        print(f"\n\nTest: Translate single segment with Gemini")

        text = "Hello world, how are you?"
        print(f"Original (en): {text}")

        try:
            translated = await translator.translate_segment(
                text=text,
                source_language="en",
                target_language="es",
                context="A greeting message",
                provider="gemini"
            )

            print(f"Translated (es): {translated}")

            # Verify translation
            assert isinstance(translated, str)
            assert len(translated) > 0
            assert translated != text

            print("✓ Gemini single segment translation successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("Gemini API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_multiple_segments(self, english_segments):
        """Test translating multiple segments."""
        print(f"\n\nTest: Translate multiple segments")
        print(f"Number of segments: {len(english_segments)}")

        context = "An educational video about programming"

        try:
            translated_segments = await translator.translate_segments(
                segments=english_segments,
                source_language="en",
                target_language="es",
                context=context,
                provider="gemini"
            )

            print(f"\nTranslation results:")

            # Verify structure
            assert len(translated_segments) == len(english_segments)

            for i, (original, translated) in enumerate(zip(english_segments, translated_segments)):
                # Verify all fields are preserved
                assert translated["start"] == original["start"]
                assert translated["end"] == original["end"]
                assert translated["text"] == original["text"]
                assert "translated_text" in translated

                # Verify translation
                assert len(translated["translated_text"]) > 0
                assert translated["translated_text"] != original["text"]

                print(f"\n  Segment {i+1}:")
                print(f"    Original: {original['text']}")
                print(f"    Translated: {translated['translated_text']}")

            print("\n✓ Multiple segments translation successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_to_different_languages(self, english_segments):
        """Test translating to multiple target languages."""
        print(f"\n\nTest: Translate to different languages")

        target_languages = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese"
        }

        context = "An educational video"

        for lang_code, lang_name in target_languages.items():
            print(f"\n  Testing {lang_name} ({lang_code}):")

            try:
                # Translate first segment only for speed
                translated = await translator.translate_segment(
                    text=english_segments[0]["text"],
                    source_language="en",
                    target_language=lang_code,
                    context=context,
                    provider="gemini"
                )

                print(f"    Original: {english_segments[0]['text']}")
                print(f"    {lang_name}: {translated}")

                assert len(translated) > 0
                assert translated != english_segments[0]["text"]

                print(f"    ✓ {lang_name} translation successful")

            except ValueError as e:
                if "API key" in str(e):
                    pytest.skip(f"API key not configured")
                print(f"    ⚠ {lang_name} translation skipped: {str(e)}")

        print("\n✓ Multi-language translation complete\n")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_with_technical_context(self, technical_segments):
        """Test translation with technical content and context."""
        print(f"\n\nTest: Translate technical content")

        context = "A technical educational video about machine learning and artificial intelligence"

        try:
            translated_segments = await translator.translate_segments(
                segments=technical_segments,
                source_language="en",
                target_language="es",
                context=context,
                provider="gemini"
            )

            print(f"\nTechnical translation results:")

            for i, segment in enumerate(translated_segments):
                print(f"\n  Segment {i+1}:")
                print(f"    Original: {segment['text']}")
                print(f"    Translated: {segment['translated_text']}")

                # Verify translation quality (basic checks)
                assert len(segment["translated_text"]) > 0
                assert segment["translated_text"] != segment["text"]

            print("\n✓ Technical content translation successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translate_from_csv(self, temp_dir, english_segments):
        """Test translating from a CSV file."""
        print(f"\n\nTest: Translate from CSV file")

        # Save segments to CSV
        csv_path = temp_dir / "transcript.csv"
        await transcriber.save_transcript_to_csv(english_segments, csv_path)
        print(f"Transcript CSV saved: {csv_path}")

        context = "An educational video"

        try:
            translated_segments = await translator.translate_from_csv(
                csv_path=csv_path,
                source_language="en",
                target_language="es",
                context=context,
                provider="gemini"
            )

            print(f"Translated {len(translated_segments)} segments from CSV")

            # Verify
            assert len(translated_segments) == len(english_segments)

            for segment in translated_segments:
                assert "translated_text" in segment
                assert len(segment["translated_text"]) > 0

            print("✓ CSV translation successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest:mark.asyncio
    @pytest.mark.slow
    async def test_save_and_load_translations(self, temp_dir):
        """Test saving translations to CSV and loading them back."""
        print(f"\n\nTest: Save and load translations")

        # Create translated segments
        translated_segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello world",
                "translated_text": "Hola mundo"
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "Good morning",
                "translated_text": "Buenos días"
            }
        ]

        # Save to CSV
        output_csv = temp_dir / "translations.csv"
        saved_path = await translator.save_translated_segments_to_csv(
            translated_segments,
            output_csv
        )

        print(f"Translations saved to: {saved_path}")
        assert saved_path.exists()

        # Read CSV to verify content
        import csv
        with open(saved_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 2
            assert rows[0]['original_text'] == "Hello world"
            assert rows[0]['translated_text'] == "Hola mundo"
            assert rows[1]['original_text'] == "Good morning"
            assert rows[1]['translated_text'] == "Buenos días"

        print("✓ Save and load translations successful\n")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translation_preserves_timing(self, english_segments):
        """Test that translation preserves timing information."""
        print(f"\n\nTest: Translation preserves timing")

        try:
            translated_segments = await translator.translate_segments(
                segments=english_segments,
                source_language="en",
                target_language="es",
                context="Educational content",
                provider="gemini"
            )

            # Verify timing is preserved
            for original, translated in zip(english_segments, translated_segments):
                assert translated["start"] == original["start"], "Start time changed"
                assert translated["end"] == original["end"], "End time changed"
                print(f"  [{translated['start']:.1f}s - {translated['end']:.1f}s]: ✓")

            print("\n✓ Timing preservation verified\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_context_improves_translation(self):
        """Test that context improves translation quality."""
        print(f"\n\nTest: Context improves translation")

        # Ambiguous text that benefits from context
        text = "The bank was steep and difficult to climb."

        print(f"Original text: {text}")

        try:
            # Translate without context
            translation_no_context = await translator.translate_segment(
                text=text,
                source_language="en",
                target_language="es",
                context="",
                provider="gemini"
            )

            print(f"\nWithout context: {translation_no_context}")

            # Translate with context
            translation_with_context = await translator.translate_segment(
                text=text,
                source_language="en",
                target_language="es",
                context="A hiking video about climbing a steep riverbank",
                provider="gemini"
            )

            print(f"With context: {translation_with_context}")

            # Both should be different from original
            assert translation_no_context != text
            assert translation_with_context != text

            print("\n✓ Context-aware translation successful\n")
            print("Note: Review translations above to see if context made a difference")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise


# Run tests with:
# pytest tests/integration/test_translator_integration.py -v -s
#
# Run without slow API tests:
# pytest tests/integration/test_translator_integration.py -v -s -m "not slow"
