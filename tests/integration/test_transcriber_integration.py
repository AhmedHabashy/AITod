"""
Integration tests for Transcriber service.

These tests verify actual transcription functionality with real audio files and API calls.
Requires:
- Valid API keys (OpenAI and/or Gemini)
- Internet connection

Run with: pytest tests/integration/test_transcriber_integration.py -v -s
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from pydub import AudioSegment
from pydub.generators import Sine

from backend.app.services.transcriber import transcriber


class TestTranscriberIntegration:
    """Integration tests for transcription functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_audio_tone(self, temp_dir):
        """Create a sample audio file with a tone for testing."""
        audio_path = temp_dir / "tone.wav"

        try:
            # Generate a 3-second sine wave at 440 Hz
            tone = Sine(440).to_audio_segment(duration=3000)
            tone.export(str(audio_path), format="wav")
            return audio_path
        except Exception as e:
            pytest.skip(f"Could not create sample audio: {str(e)}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_transcribe_audio_with_openai(self, sample_audio_tone):
        """Test transcription using OpenAI API with real audio."""
        print(f"\n\nTest: Transcribe audio with OpenAI")
        print(f"Audio path: {sample_audio_tone}")

        if not sample_audio_tone.exists():
            pytest.skip("Sample audio not available")

        try:
            segments = await transcriber.transcribe_audio(
                sample_audio_tone,
                language="en",
                provider="openai"
            )

            print(f"\nTranscription result:")
            print(f"Number of segments: {len(segments)}")

            # Verify structure
            assert isinstance(segments, list), "Segments should be a list"

            # The audio is just a tone, so transcription might be empty or minimal
            # We're mainly testing that the API call works
            if segments:
                print("\nSegments:")
                for i, segment in enumerate(segments):
                    assert "start" in segment, "Segment missing 'start' field"
                    assert "end" in segment, "Segment missing 'end' field"
                    assert "text" in segment, "Segment missing 'text' field"
                    assert isinstance(segment["start"], (int, float)), "Start time should be numeric"
                    assert isinstance(segment["end"], (int, float)), "End time should be numeric"
                    assert isinstance(segment["text"], str), "Text should be a string"

                    print(f"  [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

            print("✓ OpenAI transcription successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("OpenAI API key not configured")
            raise
        except Exception as e:
            print(f"✗ OpenAI transcription failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_transcribe_audio_with_gemini(self, sample_audio_tone):
        """Test transcription using Gemini API with real audio."""
        print(f"\n\nTest: Transcribe audio with Gemini")
        print(f"Audio path: {sample_audio_tone}")

        if not sample_audio_tone.exists():
            pytest.skip("Sample audio not available")

        try:
            segments = await transcriber.transcribe_audio(
                sample_audio_tone,
                language="en",
                provider="gemini"
            )

            print(f"\nTranscription result:")
            print(f"Number of segments: {len(segments)}")

            # Verify structure
            assert isinstance(segments, list), "Segments should be a list"

            if segments:
                print("\nSegments:")
                for i, segment in enumerate(segments):
                    assert "start" in segment, "Segment missing 'start' field"
                    assert "end" in segment, "Segment missing 'end' field"
                    assert "text" in segment, "Segment missing 'text' field"

                    print(f"  [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

            print("✓ Gemini transcription successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("Gemini API key not configured")
            raise
        except Exception as e:
            print(f"✗ Gemini transcription failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_transcribe_and_save_to_csv(self, temp_dir, sample_audio_tone):
        """Test transcribing audio and saving to CSV."""
        print(f"\n\nTest: Transcribe and save to CSV")

        if not sample_audio_tone.exists():
            pytest.skip("Sample audio not available")

        output_csv = temp_dir / "transcript.csv"

        try:
            segments, csv_path = await transcriber.transcribe_and_save(
                sample_audio_tone,
                output_csv,
                language="en",
                provider="gemini"  # Use default provider
            )

            print(f"Transcript saved to: {csv_path}")

            # Verify CSV was created
            assert csv_path.exists(), "CSV file was not created"
            assert csv_path == output_csv, "CSV path doesn't match expected path"

            # Verify segments structure
            assert isinstance(segments, list), "Segments should be a list"

            print(f"Number of segments: {len(segments)}")
            print("✓ Transcribe and save successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_transcribe_different_languages(self, sample_audio_tone):
        """Test transcription with different language codes."""
        print(f"\n\nTest: Transcribe with different languages")

        if not sample_audio_tone.exists():
            pytest.skip("Sample audio not available")

        # Test a few different languages
        languages = ["en", "es", "fr"]

        for lang in languages:
            print(f"\nTesting language: {lang}")

            try:
                segments = await transcriber.transcribe_audio(
                    sample_audio_tone,
                    language=lang,
                    provider="gemini"
                )

                print(f"  ✓ {lang} transcription successful")

            except ValueError as e:
                if "API key" in str(e):
                    pytest.skip("API key not configured")
                if "not supported" in str(e).lower():
                    print(f"  ⚠ Language {lang} not supported")
                else:
                    raise
            except Exception as e:
                print(f"  ✗ {lang} transcription failed: {str(e)}")
                raise

        print("\n✓ Multi-language test complete\n")

    @pytest.mark.asyncio
    async def test_load_and_process_transcript(self, temp_dir):
        """Test loading transcript from CSV and processing it."""
        print(f"\n\nTest: Load and process transcript")

        # Create a sample CSV
        csv_path = temp_dir / "sample_transcript.csv"
        sample_segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
            {"start": 5.0, "end": 7.5, "text": "Testing transcription"}
        ]

        # Save CSV
        await transcriber.save_transcript_to_csv(sample_segments, csv_path)
        print(f"Sample CSV created: {csv_path}")

        # Load it back
        loaded_segments = await transcriber.load_transcript_from_csv(csv_path)

        print(f"Loaded {len(loaded_segments)} segments")

        # Verify content
        assert len(loaded_segments) == len(sample_segments)

        for i, (original, loaded) in enumerate(zip(sample_segments, loaded_segments)):
            assert loaded["start"] == original["start"]
            assert loaded["end"] == original["end"]
            assert loaded["text"] == original["text"]
            print(f"  Segment {i+1}: ✓")

        # Test get full text
        full_text = transcriber.get_full_transcript_text(loaded_segments)
        print(f"\nFull transcript text:\n{full_text}")

        assert "Hello world" in full_text
        assert "This is a test" in full_text
        assert "Testing transcription" in full_text

        print("\n✓ Load and process test successful\n")


# Run tests with:
# pytest tests/integration/test_transcriber_integration.py -v -s
#
# Run without slow API tests:
# pytest tests/integration/test_transcriber_integration.py -v -s -m "not slow"
