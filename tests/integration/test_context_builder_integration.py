"""
Integration tests for ContextBuilder service.

These tests verify actual context building functionality with LLM APIs.
Requires:
- Valid API keys (OpenAI and/or Gemini)
- Internet connection

Run with: pytest tests/integration/test_context_builder_integration.py -v -s
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from backend.app.services.context_builder import context_builder
from backend.app.services.transcriber import transcriber


class TestContextBuilderIntegration:
    """Integration tests for context building functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def tech_tutorial_segments(self):
        """Sample segments from a technical tutorial."""
        return [
            {"start": 0.0, "end": 5.0, "text": "Welcome to this tutorial on machine learning."},
            {"start": 5.0, "end": 10.0, "text": "Today we will explore neural networks."},
            {"start": 10.0, "end": 15.0, "text": "Neural networks are computational models inspired by the human brain."},
            {"start": 15.0, "end": 20.0, "text": "They consist of interconnected layers of nodes called neurons."},
            {"start": 20.0, "end": 25.0, "text": "Each neuron processes input data and passes it to the next layer."},
            {"start": 25.0, "end": 30.0, "text": "Training involves adjusting weights to minimize prediction errors."}
        ]

    @pytest.fixture
    def cooking_video_segments(self):
        """Sample segments from a cooking video."""
        return [
            {"start": 0.0, "end": 3.0, "text": "Hello everyone, today we're making chocolate cake."},
            {"start": 3.0, "end": 6.0, "text": "First, preheat your oven to 350 degrees."},
            {"start": 6.0, "end": 9.0, "text": "Mix flour, sugar, cocoa powder, and baking soda."},
            {"start": 9.0, "end": 12.0, "text": "Add eggs, milk, oil, and vanilla extract."},
            {"start": 12.0, "end": 15.0, "text": "Pour the batter into a greased pan and bake for 30 minutes."}
        ]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_build_context_with_openai(self, tech_tutorial_segments):
        """Test context building using OpenAI API."""
        print(f"\n\nTest: Build context with OpenAI")
        print(f"Number of segments: {len(tech_tutorial_segments)}")

        try:
            context = await context_builder.build_context(
                tech_tutorial_segments,
                source_language="en",
                provider="openai"
            )

            print(f"\nGenerated Context:")
            print(f"{context}")
            print(f"\nContext length: {len(context)} characters")

            # Verify context was generated
            assert isinstance(context, str), "Context should be a string"
            assert len(context) > 0, "Context should not be empty"
            assert len(context) < 500, "Context should be concise (< 500 chars)"

            # Check if context contains key concepts
            context_lower = context.lower()
            # Should mention something about the topic
            has_topic_mention = any(word in context_lower for word in ["machine learning", "neural", "network", "tutorial", "brain"])

            print(f"\nContext mentions topic: {has_topic_mention}")
            print("✓ OpenAI context building successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("OpenAI API key not configured")
            raise
        except Exception as e:
            print(f"✗ OpenAI context building failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_build_context_with_gemini(self, tech_tutorial_segments):
        """Test context building using Gemini API."""
        print(f"\n\nTest: Build context with Gemini")
        print(f"Number of segments: {len(tech_tutorial_segments)}")

        try:
            context = await context_builder.build_context(
                tech_tutorial_segments,
                source_language="en",
                provider="gemini"
            )

            print(f"\nGenerated Context:")
            print(f"{context}")
            print(f"\nContext length: {len(context)} characters")

            # Verify context
            assert isinstance(context, str)
            assert len(context) > 0
            assert len(context) < 500

            # Check for topic relevance
            context_lower = context.lower()
            has_topic_mention = any(word in context_lower for word in ["machine learning", "neural", "network", "tutorial", "brain"])

            print(f"\nContext mentions topic: {has_topic_mention}")
            print("✓ Gemini context building successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("Gemini API key not configured")
            raise
        except Exception as e:
            print(f"✗ Gemini context building failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_build_context_different_topics(self, cooking_video_segments):
        """Test context building with different content types."""
        print(f"\n\nTest: Build context for cooking video")

        try:
            context = await context_builder.build_context(
                cooking_video_segments,
                source_language="en",
                provider="gemini"
            )

            print(f"\nGenerated Context:")
            print(f"{context}")

            # Verify context
            assert isinstance(context, str)
            assert len(context) > 0

            # Check for cooking/recipe related terms
            context_lower = context.lower()
            has_cooking_terms = any(word in context_lower for word in ["cook", "recipe", "bake", "cake", "food", "kitchen"])

            print(f"\nContext mentions cooking: {has_cooking_terms}")
            print("✓ Different topic context building successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_build_context_from_csv_file(self, temp_dir, tech_tutorial_segments):
        """Test building context from a CSV file."""
        print(f"\n\nTest: Build context from CSV file")

        # Save segments to CSV
        csv_path = temp_dir / "transcript.csv"
        await transcriber.save_transcript_to_csv(tech_tutorial_segments, csv_path)
        print(f"CSV saved to: {csv_path}")

        try:
            context = await context_builder.build_context_from_file(
                csv_path,
                source_language="en",
                provider="gemini"
            )

            print(f"\nGenerated Context from CSV:")
            print(f"{context}")

            # Verify context
            assert isinstance(context, str)
            assert len(context) > 0

            print("✓ Context building from CSV successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    async def test_build_simple_context_functionality(self, tech_tutorial_segments):
        """Test the simple context builder (no API required)."""
        print(f"\n\nTest: Build simple context (no API)")

        context = context_builder.build_simple_context(tech_tutorial_segments)

        print(f"\nSimple Context:")
        print(f"{context}")

        # Verify structure
        assert isinstance(context, str)
        assert len(context) > 0

        # Should contain duration and word count info
        assert "duration" in context.lower() or "seconds" in context.lower()
        assert "word" in context.lower()

        print("✓ Simple context building successful\n")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_context_comparison_simple_vs_llm(self, tech_tutorial_segments):
        """Compare simple context vs LLM-generated context."""
        print(f"\n\nTest: Compare simple vs LLM context")

        # Build simple context
        simple_context = context_builder.build_simple_context(tech_tutorial_segments)
        print(f"\nSimple Context ({len(simple_context)} chars):")
        print(f"{simple_context}\n")

        try:
            # Build LLM context
            llm_context = await context_builder.build_context(
                tech_tutorial_segments,
                source_language="en",
                provider="gemini"
            )
            print(f"LLM Context ({len(llm_context)} chars):")
            print(f"{llm_context}\n")

            # Compare
            print("Comparison:")
            print(f"  Simple length: {len(simple_context)} chars")
            print(f"  LLM length: {len(llm_context)} chars")
            print(f"  LLM is more concise: {len(llm_context) < len(simple_context)}")

            print("\n✓ Context comparison successful\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_context_for_translation(self, tech_tutorial_segments):
        """Test that context is suitable for translation purposes."""
        print(f"\n\nTest: Context quality for translation")

        try:
            context = await context_builder.build_context(
                tech_tutorial_segments,
                source_language="en",
                provider="gemini"
            )

            print(f"\nContext for translation:")
            print(f"{context}\n")

            # Verify context characteristics for translation
            assert len(context) > 10, "Context too short to be useful"
            assert len(context) < 500, "Context too long for efficient translation"

            # Check if it's descriptive (not just metadata)
            words = context.split()
            assert len(words) >= 5, "Context should have meaningful content"

            print("✓ Context is suitable for translation\n")

        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not configured")
            raise


# Run tests with:
# pytest tests/integration/test_context_builder_integration.py -v -s
#
# Run without slow API tests:
# pytest tests/integration/test_context_builder_integration.py -v -s -m "not slow"
