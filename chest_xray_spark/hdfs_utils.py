"""
HDFS Utilities for Reading and Writing Files

Uses HDFS CLI commands for direct HDFS access.
Falls back to local filesystem operations for development.
"""
import os
import json
import logging
import subprocess
import tempfile
from typing import Optional, Union
import io

from .config import (
    HDFS_NAMENODE,
    HDFS_BASE_PATH,
    HDFS_UPLOADS_PATH,
    HDFS_RESULTS_PATH,
    HDFS_MODEL_PATH
)

logger = logging.getLogger(__name__)

# Path to hdfs command
HDFS_CMD = os.environ.get("HDFS_CMD", "/Users/aditchauhan/hadoop/bin/hdfs")


def use_local_fs():
    """Check if we should use local filesystem instead of HDFS."""
    return os.environ.get("USE_LOCAL_FS", "true").lower() == "true"


def run_hdfs_cmd(args: list) -> tuple:
    """
    Run an HDFS command and return (success, output).
    
    Args:
        args: List of arguments for hdfs dfs command
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        cmd = [HDFS_CMD, "dfs"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)


def hdfs_available() -> bool:
    """Check if HDFS is available and accessible."""
    if use_local_fs():
        return False
    
    success, _ = run_hdfs_cmd(["-ls", "/"])
    return success


def get_local_path(hdfs_path: str) -> str:
    """
    Convert HDFS path to local path for development.
    
    Args:
        hdfs_path: HDFS path like /chest_xray/results/...
        
    Returns:
        Local filesystem path
    """
    from .config import BASE_DIR
    # Remove leading slash and replace with local base
    local_path = os.path.join(BASE_DIR, hdfs_path.lstrip('/'))
    return local_path


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists (works with HDFS or local).
    
    Args:
        path: Directory path
    """
    if not hdfs_available():
        # Local filesystem
        local_path = get_local_path(path) if path.startswith('/') else path
        os.makedirs(local_path, exist_ok=True)
    else:
        # HDFS - create directory
        success, output = run_hdfs_cmd(["-mkdir", "-p", path])
        if not success:
            logger.debug(f"HDFS mkdir result: {output}")


def save_bytes_to_hdfs(data: bytes, hdfs_path: str) -> str:
    """
    Save bytes to HDFS or local filesystem.
    
    Args:
        data: Bytes to save
        hdfs_path: Target path
        
    Returns:
        Actual path where file was saved
    """
    if not hdfs_available():
        # Local filesystem fallback
        local_path = get_local_path(hdfs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        logger.info(f"Saved to local: {local_path}")
        return local_path
    
    # HDFS - write via temp file and put
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(hdfs_path)
        run_hdfs_cmd(["-mkdir", "-p", parent_dir])
        
        # Upload file
        success, output = run_hdfs_cmd(["-put", "-f", tmp_path, hdfs_path])
        if success:
            logger.info(f"Saved to HDFS: {hdfs_path}")
            return hdfs_path
        else:
            logger.error(f"HDFS put failed: {output}")
            # Fallback to local
            local_path = get_local_path(hdfs_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)
            logger.info(f"Fallback to local: {local_path}")
            return local_path
    finally:
        os.unlink(tmp_path)


def save_image_to_hdfs(image_bytes: bytes, hdfs_path: str) -> str:
    """
    Save image bytes to HDFS.
    
    Args:
        image_bytes: Raw image bytes
        hdfs_path: Target HDFS path
        
    Returns:
        Actual path where image was saved
    """
    return save_bytes_to_hdfs(image_bytes, hdfs_path)


def save_json_to_hdfs(data: dict, hdfs_path: str) -> str:
    """
    Save JSON data to HDFS.
    
    Args:
        data: Dictionary to save as JSON
        hdfs_path: Target HDFS path
        
    Returns:
        Actual path where JSON was saved
    """
    json_bytes = json.dumps(data, indent=2).encode('utf-8')
    return save_bytes_to_hdfs(json_bytes, hdfs_path)


def read_bytes_from_hdfs(hdfs_path: str) -> bytes:
    """
    Read bytes from HDFS or local filesystem.
    
    Args:
        hdfs_path: Source path
        
    Returns:
        File contents as bytes
    """
    if not hdfs_available():
        # Local filesystem
        local_path = get_local_path(hdfs_path)
        with open(local_path, 'rb') as f:
            return f.read()
    
    # HDFS - read via cat to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        success, output = run_hdfs_cmd(["-get", hdfs_path, tmp_path])
        if success:
            with open(tmp_path, 'rb') as f:
                return f.read()
        else:
            raise IOError(f"Failed to read from HDFS: {output}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def read_json_from_hdfs(hdfs_path: str) -> dict:
    """
    Read JSON from HDFS.
    
    Args:
        hdfs_path: Source HDFS path
        
    Returns:
        Parsed JSON as dictionary
    """
    data = read_bytes_from_hdfs(hdfs_path)
    return json.loads(data.decode('utf-8'))


def list_directory(hdfs_path: str) -> list:
    """
    List contents of HDFS directory.
    
    Args:
        hdfs_path: Directory path
        
    Returns:
        List of file/directory names
    """
    if not hdfs_available():
        # Local filesystem
        local_path = get_local_path(hdfs_path)
        if os.path.exists(local_path):
            return os.listdir(local_path)
        return []
    
    # HDFS - use ls command
    success, output = run_hdfs_cmd(["-ls", hdfs_path])
    if not success:
        return []
    
    # Parse ls output - each line has format: drwxr-xr-x   - user group size date time path
    files = []
    for line in output.strip().split('\n'):
        if line.startswith('Found') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 8:
            path = parts[-1]
            files.append(os.path.basename(path))
    
    return files


def create_result_directory(patient_id: str, timestamp: str) -> str:
    """
    Create a result directory for a patient.
    
    Args:
        patient_id: Patient identifier
        timestamp: Timestamp string
        
    Returns:
        Path to the created directory
    """
    dir_name = f"{patient_id}_{timestamp}"
    dir_path = f"{HDFS_RESULTS_PATH}/{dir_name}"
    
    ensure_directory(dir_path)
    
    return dir_path


def save_inference_results(
    patient_id: str,
    timestamp: str,
    original_image: bytes,
    predictions: dict,
    metadata: dict,
    gradcam_images: dict = None
) -> str:
    """
    Save complete inference results to HDFS.
    
    Args:
        patient_id: Patient identifier
        timestamp: Timestamp string
        original_image: Original uploaded image bytes
        predictions: Prediction probabilities
        metadata: Patient metadata
        gradcam_images: Dictionary of label -> overlay image bytes
        
    Returns:
        Path to results directory
    """
    import numpy as np
    from PIL import Image
    
    # Create result directory
    result_dir = create_result_directory(patient_id, timestamp)
    
    # Save original image
    save_image_to_hdfs(original_image, f"{result_dir}/original.png")
    
    # Save predictions
    save_json_to_hdfs(predictions, f"{result_dir}/predictions.json")
    
    # Save metadata
    save_json_to_hdfs(metadata, f"{result_dir}/metadata.json")
    
    # Save GRAD-CAM images
    if gradcam_images:
        for label, overlay_array in gradcam_images.items():
            # Convert numpy array to PNG bytes
            if isinstance(overlay_array, np.ndarray):
                img = Image.fromarray(overlay_array.astype(np.uint8))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                overlay_bytes = img_buffer.getvalue()
            else:
                overlay_bytes = overlay_array
            
            # Sanitize label for filename
            safe_label = label.replace(' ', '_').replace('/', '_')
            save_image_to_hdfs(overlay_bytes, f"{result_dir}/gradcam_{safe_label}.png")
    
    logger.info(f"Saved inference results to: {result_dir}")
    return result_dir


def get_result_files(result_dir: str) -> dict:
    """
    Get all files from a result directory.
    
    Args:
        result_dir: Path to result directory
        
    Returns:
        Dictionary with file paths and contents info
    """
    files = list_directory(result_dir)
    
    result = {
        'directory': result_dir,
        'files': {}
    }
    
    for fname in files:
        file_path = f"{result_dir}/{fname}"
        if fname.endswith('.json'):
            result['files'][fname] = read_json_from_hdfs(file_path)
        else:
            result['files'][fname] = file_path
    
    return result
