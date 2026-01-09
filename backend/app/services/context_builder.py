from typing import List, Dict, Optional

from app.utils import llm_client, settings


class ContextBuilder:
    """Service for building context summaries from transcript segments."""

    @staticmethod
    async def build_context(
        segments: List[Dict[str, any]],
        source_language: str = "en",
        provider: Optional[str] = None
    ) -> str:
        """Build a context summary from transcript segments.

        This context will be used to improve translation quality by providing
        information about the overall content, topic, and key terms.

        Args:
            segments: List of transcript segments with 'start', 'end', 'text'
            source_language: Source language code (ISO 639-1)
            provider: LLM provider to use ('openai' or 'gemini'). If None, uses default.

        Returns:
            Context summary as a string

        Raises:
            ValueError: If segments is empty or invalid
            Exception: If context building fails
        """
        if not segments:
            raise ValueError("Cannot build context from empty segments")

        # Extract full text from segments
        full_text = " ".join(segment['text'] for segment in segments)

        # If text is too short, return a simple context
        if len(full_text.strip()) < 10:
            return "Short video content without specific context."

        # Build context using LLM
        context = await ContextBuilder._generate_context_with_llm(
            full_text=full_text,
            source_language=source_language,
            provider=provider
        )

        return context

    @staticmethod
    async def _generate_context_with_llm(
        full_text: str,
        source_language: str,
        provider: Optional[str] = None
    ) -> str:
        """Generate context summary using LLM.

        Args:
            full_text: Complete transcript text
            source_language: Source language code
            provider: LLM provider to use

        Returns:
            Context summary string
        """
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        # Create prompt for context generation
        prompt = f"""Analyze the following transcript in {source_language} language and provide a brief context summary (2-3 sentences maximum).

Include:
1. The main topic or theme
2. The tone (e.g., formal, casual, technical, educational)
3. Any key technical terms, proper nouns, or domain-specific vocabulary

Keep it concise and focused on information that would help a translator maintain accuracy and consistency.

Transcript:
"{full_text[:2000]}"

Context Summary:"""

        try:
            # Use the translation method from llm_client with identity translation to get summary
            # We'll use a workaround: call the LLM directly through the internal methods
            if provider == "openai":
                context = await llm_client._translate_with_openai(
                    text=prompt,
                    source_language=source_language,
                    target_language="en",
                    context=""
                )
            elif provider == "gemini":
                context = await llm_client._translate_with_gemini(
                    text=prompt,
                    source_language=source_language,
                    target_language="en",
                    context=""
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return context.strip()

        except Exception as e:
            # Fallback to basic context if LLM fails
            return f"Content in {source_language} language. Length: approximately {len(full_text.split())} words."

    @staticmethod
    def build_simple_context(
        segments: List[Dict[str, any]],
        source_language: str = "en"
    ) -> str:
        """Build a simple context without using LLM.

        This is a fallback method that creates basic context information
        without making API calls.

        Args:
            segments: List of transcript segments
            source_language: Source language code

        Returns:
            Simple context summary string
        """
        if not segments:
            return "Empty transcript."

        # Extract full text
        full_text = " ".join(segment['text'] for segment in segments)
        word_count = len(full_text.split())
        duration = segments[-1]['end'] - segments[0]['start'] if segments else 0

        # Build basic context
        context = (
            f"Video transcript in {source_language} language. "
            f"Duration: {duration:.1f} seconds. "
            f"Approximate word count: {word_count} words."
        )

        return context

    @staticmethod
    async def build_context_from_file(
        csv_path,
        source_language: str = "en",
        provider: Optional[str] = None
    ) -> str:
        """Build context from a CSV transcript file.

        Args:
            csv_path: Path to the CSV transcript file
            source_language: Source language code
            provider: LLM provider to use

        Returns:
            Context summary string

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            Exception: If context building fails
        """
        # Import here to avoid circular dependency
        from .transcriber import transcriber

        # Load transcript from CSV
        segments = await transcriber.load_transcript_from_csv(csv_path)

        # Build context
        context = await ContextBuilder.build_context(
            segments=segments,
            source_language=source_language,
            provider=provider
        )

        return context


# Create singleton instance
context_builder = ContextBuilder()
