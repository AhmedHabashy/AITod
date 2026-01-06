import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException

from .config import settings


class FileHandler:
    """Handles file operations for video uploads and processing."""

    @staticmethod
    def generate_file_id() -> str:
        """Generate a unique file ID using UUID4.

        Returns:
            A unique string identifier
        """
        return str(uuid.uuid4())

    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """Validate if the file type is allowed.

        Args:
            filename: Name of the file to validate

        Returns:
            True if file type is allowed, False otherwise
        """
        file_extension = filename.split(".")[-1].lower()
        return file_extension in settings.ALLOWED_VIDEO_FORMATS

    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate if the file size is within limits.

        Args:
            file_size: Size of the file in bytes

        Returns:
            True if file size is acceptable, False otherwise
        """
        return file_size <= settings.MAX_FILE_SIZE_BYTES

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Extract file extension from filename.

        Args:
            filename: Name of the file

        Returns:
            File extension without the dot (e.g., 'mp4')
        """
        return filename.split(".")[-1].lower()

    @staticmethod
    async def save_upload_file(
        upload_file: UploadFile,
        file_id: Optional[str] = None
    ) -> Tuple[str, Path]:
        """Save an uploaded file to the upload directory.

        Args:
            upload_file: FastAPI UploadFile object
            file_id: Optional file ID, will generate new one if not provided

        Returns:
            Tuple of (file_id, file_path)

        Raises:
            HTTPException: If file validation fails
        """
        # Validate file type
        if not FileHandler.validate_file_type(upload_file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_VIDEO_FORMATS)}"
            )

        # Generate file ID if not provided
        if file_id is None:
            file_id = FileHandler.generate_file_id()

        # Get file extension
        file_extension = FileHandler.get_file_extension(upload_file.filename)

        # Create file path
        file_path = settings.UPLOAD_DIR / f"{file_id}.{file_extension}"

        # Read and validate file size
        content = await upload_file.read()
        if not FileHandler.validate_file_size(len(content)):
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
            )

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        return file_id, file_path

    @staticmethod
    async def delete_file(file_path: Path) -> bool:
        """Delete a file from the filesystem.

        Args:
            file_path: Path to the file to delete

        Returns:
            True if file was deleted, False if file didn't exist
        """
        try:
            if file_path.exists():
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False

    @staticmethod
    async def cleanup_files(file_id: str) -> None:
        """Clean up all files associated with a file ID.

        Removes video, audio, transcript, and output files.

        Args:
            file_id: The file ID to clean up
        """
        # Patterns to search for
        patterns = [
            settings.UPLOAD_DIR / f"{file_id}.*",
            settings.AUDIO_DIR / f"{file_id}.*",
            settings.TRANSCRIPT_DIR / f"{file_id}.*",
            settings.OUTPUT_DIR / f"{file_id}.*",
        ]

        for pattern_path in patterns:
            # Get parent directory and filename pattern
            parent_dir = pattern_path.parent
            file_pattern = pattern_path.name

            # Find and delete matching files
            for file_path in parent_dir.glob(file_pattern):
                await FileHandler.delete_file(file_path)

    @staticmethod
    def get_file_path(file_id: str, storage_type: str, extension: str) -> Path:
        """Get the full path for a file.

        Args:
            file_id: The file ID
            storage_type: Type of storage ('upload', 'audio', 'transcript', 'output')
            extension: File extension without dot (e.g., 'mp4', 'wav')

        Returns:
            Path object for the file
        """
        storage_dir = settings.get_storage_path(storage_type)
        return storage_dir / f"{file_id}.{extension}"

    @staticmethod
    def file_exists(file_path: Path) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        return file_path.exists() and file_path.is_file()


# Create a singleton instance for easy access
file_handler = FileHandler()
