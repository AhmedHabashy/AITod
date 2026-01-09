"""
Integration tests for SRTGenerator service.

These tests verify actual SRT generation functionality with realistic data.
No API calls required.

Run with: pytest tests/integration/test_srt_generator_integration.py -v -s
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import srt

from backend.app.services.srt_generator import srt_generator
from backend.app.services.transcriber import transcriber
from backend.app.services.translator import translator


class TestSRTGeneratorIntegration:
    """Integration tests for SRT generation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def realistic_translated_segments(self):
        """Realistic translated segments from a video."""
        return [
            {
                "start": 0.0,
                "end": 4.5,
                "text": "Welcome to this comprehensive tutorial on Python programming.",
                "translated_text": "Bienvenido a este tutorial completo sobre programación en Python."
            },
            {
                "start": 4.5,
                "end": 9.2,
                "text": "In this video, we'll explore the fundamental concepts of programming.",
                "translated_text": "En este video, exploraremos los conceptos fundamentales de la programación."
            },
            {
                "start": 9.2,
                "end": 14.8,
                "text": "Python is known for its simplicity and readability, making it perfect for beginners.",
                "translated_text": "Python es conocido por su simplicidad y legibilidad, lo que lo hace perfecto para principiantes."
            },
            {
                "start": 14.8,
                "end": 19.5,
                "text": "Let's start by understanding variables and data types.",
                "translated_text": "Comencemos entendiendo las variables y los tipos de datos."
            },
            {
                "start": 19.5,
                "end": 24.3,
                "text": "Variables are containers for storing data values in your program.",
                "translated_text": "Las variables son contenedores para almacenar valores de datos en tu programa."
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_professional_srt_from_translations(self, temp_dir, realistic_translated_segments):
        """Test generating a professional SRT file from real translated segments."""
        print(f"\n\nTest: Generate professional SRT from translations")
        print(f"Number of segments: {len(realistic_translated_segments)}")

        output_path = temp_dir / "professional_subtitles.srt"

        # Generate SRT
        result_path = await srt_generator.save_srt(
            realistic_translated_segments,
            output_path,
            use_translated=True
        )

        print(f"\nSRT file generated: {result_path}")
        assert result_path.exists()

        # Read and verify SRT content
        with open(result_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        print(f"\nGenerated SRT content:\n")
        print("=" * 60)
        print(srt_content)
        print("=" * 60)

        # Parse SRT to verify it's valid
        parsed_subs = list(srt.parse(srt_content))
        assert len(parsed_subs) == len(realistic_translated_segments)

        # Verify each subtitle
        for i, (sub, original) in enumerate(zip(parsed_subs, realistic_translated_segments)):
            assert sub.index == i + 1
            assert abs(sub.start.total_seconds() - original["start"]) < 0.001
            assert abs(sub.end.total_seconds() - original["end"]) < 0.001
            assert sub.content == original["translated_text"]

        print(f"\n✓ Professional SRT generation successful")
        print(f"  Total duration: {realistic_translated_segments[-1]['end']:.1f}s")
        print(f"  Subtitles: {len(parsed_subs)}")
        print()

    @pytest.mark.asyncio
    async def test_generate_bilingual_srt(self, temp_dir, realistic_translated_segments):
        """Test generating SRT files for both original and translated text."""
        print(f"\n\nTest: Generate bilingual SRT files")

        # Generate original language SRT
        original_path = temp_dir / "original_en.srt"
        await srt_generator.save_srt(
            realistic_translated_segments,
            original_path,
            use_translated=False
        )

        # Generate translated language SRT
        translated_path = temp_dir / "translated_es.srt"
        await srt_generator.save_srt(
            realistic_translated_segments,
            translated_path,
            use_translated=True
        )

        print(f"Original SRT: {original_path}")
        print(f"Translated SRT: {translated_path}")

        # Verify both exist
        assert original_path.exists()
        assert translated_path.exists()

        # Compare them
        with open(original_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        with open(translated_path, 'r', encoding='utf-8') as f:
            translated_content = f.read()

        print(f"\nOriginal SRT (first 200 chars):\n{original_content[:200]}...")
        print(f"\nTranslated SRT (first 200 chars):\n{translated_content[:200]}...")

        # Verify they're different
        assert original_content != translated_content

        # Both should have same number of subtitles
        original_subs = list(srt.parse(original_content))
        translated_subs = list(srt.parse(translated_content))
        assert len(original_subs) == len(translated_subs)

        print(f"\n✓ Bilingual SRT generation successful\n")

    @pytest.mark.asyncio
    async def test_generate_srt_from_csv_workflow(self, temp_dir):
        """Test the full CSV to SRT workflow."""
        print(f"\n\nTest: Full CSV to SRT workflow")

        # Step 1: Create transcript segments
        transcript_segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello everyone"},
            {"start": 3.0, "end": 6.0, "text": "Welcome to this video"},
            {"start": 6.0, "end": 9.0, "text": "Let's get started"}
        ]

        # Step 2: Save to CSV
        csv_path = temp_dir / "transcript.csv"
        await transcriber.save_transcript_to_csv(transcript_segments, csv_path)
        print(f"1. Transcript saved to CSV: {csv_path}")

        # Step 3: Create translated CSV
        translated_segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello everyone", "translated_text": "Hola a todos"},
            {"start": 3.0, "end": 6.0, "text": "Welcome to this video", "translated_text": "Bienvenidos a este video"},
            {"start": 6.0, "end": 9.0, "text": "Let's get started", "translated_text": "Empecemos"}
        ]

        translation_csv_path = temp_dir / "translation.csv"
        await translator.save_translated_segments_to_csv(translated_segments, translation_csv_path)
        print(f"2. Translation saved to CSV: {translation_csv_path}")

        # Step 4: Generate SRT from translation CSV
        srt_path = temp_dir / "subtitles.srt"
        result_path = await srt_generator.generate_srt_from_csv(
            csv_path=translation_csv_path,
            output_path=srt_path,
            use_translated=True
        )

        print(f"3. SRT generated: {result_path}")

        # Verify SRT
        assert result_path.exists()

        with open(result_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        print(f"\nGenerated SRT:\n{srt_content}")

        # Verify content
        assert "Hola a todos" in srt_content
        assert "Bienvenidos a este video" in srt_content
        assert "Empecemos" in srt_content

        print(f"\n✓ CSV to SRT workflow successful\n")

    @pytest.mark.asyncio
    async def test_srt_with_special_characters(self, temp_dir):
        """Test SRT generation with special characters and accents."""
        print(f"\n\nTest: SRT with special characters")

        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Test with special chars",
                "translated_text": "¡Hola! ¿Cómo estás?"
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "More special chars",
                "translated_text": "Música, café, años, niño"
            },
            {
                "start": 6.0,
                "end": 9.0,
                "text": "Symbols test",
                "translated_text": "Test: 100%, $50, €20, @user, #tag"
            }
        ]

        output_path = temp_dir / "special_chars.srt"
        await srt_generator.save_srt(segments, output_path, use_translated=True)

        # Verify file was created and contains special characters
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\nSRT with special characters:\n{content}")

        # Verify special characters are preserved
        assert "¡Hola!" in content
        assert "¿Cómo estás?" in content
        assert "café" in content
        assert "@user" in content
        assert "#tag" in content

        print(f"\n✓ Special characters handled correctly\n")

    @pytest.mark.asyncio
    async def test_srt_load_and_verify(self, temp_dir, realistic_translated_segments):
        """Test saving SRT and loading it back for verification."""
        print(f"\n\nTest: Save and load SRT")

        # Save SRT
        srt_path = temp_dir / "test.srt"
        await srt_generator.save_srt(realistic_translated_segments, srt_path, use_translated=True)
        print(f"SRT saved: {srt_path}")

        # Load it back
        loaded_segments = await srt_generator.load_srt(srt_path)
        print(f"Loaded {len(loaded_segments)} segments")

        # Verify loaded segments match original
        assert len(loaded_segments) == len(realistic_translated_segments)

        for i, (loaded, original) in enumerate(zip(loaded_segments, realistic_translated_segments)):
            assert abs(loaded["start"] - original["start"]) < 0.001
            assert abs(loaded["end"] - original["end"]) < 0.001
            assert loaded["text"] == original["translated_text"]
            print(f"  Segment {i+1}: ✓")

        print(f"\n✓ Save and load verification successful\n")

    @pytest.mark.asyncio
    async def test_srt_timing_accuracy(self, temp_dir):
        """Test that SRT timing is accurately formatted."""
        print(f"\n\nTest: SRT timing accuracy")

        # Test various timing edge cases
        segments = [
            {"start": 0.0, "end": 0.5, "text": "Very short", "translated_text": "Muy corto"},
            {"start": 0.5, "end": 60.0, "text": "One minute", "translated_text": "Un minuto"},
            {"start": 60.0, "end": 125.5, "text": "Over a minute", "translated_text": "Más de un minuto"},
            {"start": 3600.0, "end": 3665.123, "text": "One hour mark", "translated_text": "Marca de una hora"}
        ]

        srt_path = temp_dir / "timing_test.srt"
        await srt_generator.save_srt(segments, srt_path, use_translated=True)

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\nTiming test SRT:\n{content}")

        # Verify specific timestamps
        assert "00:00:00,000 --> 00:00:00,500" in content  # 0.0 to 0.5s
        assert "00:01:00,000 --> 00:02:05,500" in content  # 60s to 125.5s
        assert "01:00:00,000 --> 01:01:05,123" in content  # 3600s to 3665.123s

        print(f"\n✓ Timing accuracy verified\n")

    @pytest.mark.asyncio
    async def test_real_video_subtitle_generation(self, temp_dir):
        """Simulate generating subtitles for a real video scenario."""
        print(f"\n\nTest: Real video subtitle generation scenario")

        # Simulate a 2-minute video with 20 segments
        segments = []
        for i in range(20):
            start = i * 6.0
            end = start + 5.5
            segments.append({
                "start": start,
                "end": end,
                "text": f"Original sentence number {i+1} in English.",
                "translated_text": f"Oración original número {i+1} en inglés."
            })

        print(f"Generating subtitles for a {end:.1f}s video with {len(segments)} segments")

        # Generate SRT
        srt_path = temp_dir / "real_video_subs.srt"
        await srt_generator.save_srt(segments, srt_path, use_translated=True)

        # Verify
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        parsed_subs = list(srt.parse(content))
        assert len(parsed_subs) == 20

        print(f"\nFirst 3 subtitles:")
        for sub in parsed_subs[:3]:
            print(f"  {sub.index}: [{sub.start} --> {sub.end}] {sub.content}")

        print(f"\nLast subtitle:")
        last_sub = parsed_subs[-1]
        print(f"  {last_sub.index}: [{last_sub.start} --> {last_sub.end}] {last_sub.content}")

        print(f"\n✓ Real video scenario successful")
        print(f"  Video duration: {end:.1f}s")
        print(f"  Total subtitles: {len(parsed_subs)}")
        print(f"  File size: {srt_path.stat().st_size} bytes")
        print()


# Run tests with:
# pytest tests/integration/test_srt_generator_integration.py -v -s
