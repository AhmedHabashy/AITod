import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application configuration settings loaded from environment variables."""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # FastAPI Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Project Root Directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # File Storage Paths
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
    AUDIO_DIR: Path = BASE_DIR / os.getenv("AUDIO_DIR", "storage/audio")
    TRANSCRIPT_DIR: Path = BASE_DIR / os.getenv("TRANSCRIPT_DIR", "storage/transcripts")
    OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "storage/output")

    # File Upload Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_VIDEO_FORMATS: List[str] = os.getenv(
        "ALLOWED_VIDEO_FORMATS",
        "mp4,avi,mov,mkv,webm"
    ).split(",")

    # Processing Configuration
    CHUNK_SIZE_SECONDS: int = int(os.getenv("CHUNK_SIZE_SECONDS", "60"))
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "5"))

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-audio-preview")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # Frontend Configuration
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")

    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = os.getenv(
        "SUPPORTED_LANGUAGES",
        "en,zh,hi,es,ar,fr,bn,pt,ru,ur,id,de,ja,mr,te,tr,ta,vi,ko,sw"
    ).split(",")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration settings."""
        errors = []

        # Check if at least one LLM API key is provided
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            errors.append("At least one LLM API key (OPENAI_API_KEY or GEMINI_API_KEY) must be set")

        # Validate default provider
        if cls.DEFAULT_LLM_PROVIDER not in ["openai", "gemini"]:
            errors.append(f"DEFAULT_LLM_PROVIDER must be 'openai' or 'gemini', got '{cls.DEFAULT_LLM_PROVIDER}'")

        # Validate that the default provider has an API key
        if cls.DEFAULT_LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when DEFAULT_LLM_PROVIDER is 'openai'")

        if cls.DEFAULT_LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required when DEFAULT_LLM_PROVIDER is 'gemini'")

        # Ensure storage directories exist
        for directory in [cls.UPLOAD_DIR, cls.AUDIO_DIR, cls.TRANSCRIPT_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))

    @classmethod
    def get_storage_path(cls, storage_type: str) -> Path:
        """Get the path for a specific storage type.

        Args:
            storage_type: One of 'upload', 'audio', 'transcript', 'output'

        Returns:
            Path object for the requested storage directory
        """
        storage_map = {
            "upload": cls.UPLOAD_DIR,
            "audio": cls.AUDIO_DIR,
            "transcript": cls.TRANSCRIPT_DIR,
            "output": cls.OUTPUT_DIR
        }

        if storage_type not in storage_map:
            raise ValueError(f"Invalid storage type: {storage_type}. Must be one of {list(storage_map.keys())}")

        return storage_map[storage_type]


# Create a singleton instance
settings = Settings()

# Validate configuration on import
if __name__ != "__main__":
    try:
        settings.validate()
    except ValueError as e:
        print(f"Warning: {e}")
