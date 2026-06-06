import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utilities:
    @staticmethod
    def get_list_of_file_names_from_directory(directory_path: str) -> List[str]:
        """
        Retrieves a list of filenames from the specified directory.
        
        Args:
            directory_path (str): The local path to scan.
            
        Returns:
            List[str]: A list of filenames found.
        """
        try:
            path = Path(directory_path)
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a valid directory: {directory_path}")
            
            # Using glob to filter by extension efficiently
            files = [f.name for f in path]
            logger.info(f"Found {len(files)} files in {directory_path}")
            return files
        except Exception as e:
            logger.error(f"Root cause analysis: Failed to list directory {directory_path}. Error: {e}")
            raise

    @staticmethod
    def get_local_file_metadata(file_path: str) -> Dict[str, Any]:
        """
        Retrieves system-level metadata for a local file.
        
        Args:
            file_path (str): The absolute or relative path to the file.
            
        Returns:
            Dict[str, Any]: File statistics (size, timestamps).
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stats = path.stat()
            return {
                "file_name": path.name,
                "file_size_bytes": stats.st_size,
                "created_at": stats.st_ctime,
                "modified_at": stats.st_mtime,
                "absolute_path": str(path.absolute())
            }
        except Exception as e:
            logger.error(f"Root cause analysis: Could not retrieve metadata for {file_path}. Error: {e}")
            raise

    @staticmethod
    def verify_processing_status(file_path: str) -> bool:
        """
        Verifies if a file exists and is accessible. 
        Replaces async job polling logic for local synchronous workflows.
        
        Args:
            file_path (str): The path to the processed output.
            
        Returns:
            bool: True if file exists and is ready for use.
        """
        path = Path(file_path)
        is_ready = path.exists() and path.is_file()
        logger.info(f"Verification check for {file_path}: {'Success' if is_ready else 'Pending'}")
        return is_ready