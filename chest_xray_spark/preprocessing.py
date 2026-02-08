"""
Image and metadata preprocessing for Chest X-Ray Inference

Matches the preprocessing from the training notebook:
- Image: 256x256 RGB with DenseNet preprocessing
- Metadata: age_norm, sex_bin, view_AP, view_PA
"""
import io
import numpy as np
from PIL import Image
import tensorflow as tf

from .config import IMAGE_SIZE, IMAGE_CHANNELS, META_COLS


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes for model inference.
    
    Args:
        image_bytes: Raw image bytes (PNG or JPEG)
        
    Returns:
        Preprocessed image tensor of shape (1, 256, 256, 3)
    """
    # Decode image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply DenseNet preprocessing (scale to [-1, 1] range)
    # keras.applications.densenet.preprocess_input does: x /= 127.5; x -= 1.
    img_array = img_array / 127.5 - 1.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


def preprocess_image_path(image_path: str) -> np.ndarray:
    """
    Preprocess an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor of shape (1, 256, 256, 3)
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return preprocess_image_bytes(image_bytes)


def preprocess_image_tf(image_path: str) -> tf.Tensor:
    """
    TensorFlow-native image preprocessing (for Spark batch processing).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor of shape (256, 256, 3)
    """
    # Read and decode image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=IMAGE_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    
    # Apply DenseNet preprocessing
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1.0
    
    return img


def preprocess_metadata(age: int, gender: str, view: str) -> np.ndarray:
    """
    Preprocess patient metadata for model inference.
    
    Args:
        age: Patient age in years
        gender: 'M' or 'F'
        view: 'AP' or 'PA'
        
    Returns:
        Metadata array of shape (1, 4): [age_norm, sex_bin, view_AP, view_PA]
    """
    # Normalize age (divide by 100 as in training)
    # Handle unrealistic ages (cap at 100)
    age = min(max(0, age), 100)
    age_norm = age / 100.0
    
    # Encode gender (1 for Male, 0 for Female)
    sex_bin = 1.0 if gender.upper() == 'M' else 0.0
    
    # One-hot encode view position
    view = view.upper()
    view_AP = 1.0 if view == 'AP' else 0.0
    view_PA = 1.0 if view == 'PA' else 0.0
    
    # Create metadata array
    metadata = np.array([[age_norm, sex_bin, view_AP, view_PA]], dtype=np.float32)
    
    return metadata


def validate_input(image_bytes: bytes, age: int, gender: str, view: str) -> tuple:
    """
    Validate and preprocess all inputs.
    
    Args:
        image_bytes: Raw image bytes
        age: Patient age
        gender: Patient gender
        view: View position
        
    Returns:
        Tuple of (image_batch, metadata_batch, errors)
        errors is None if valid, otherwise contains error messages
    """
    errors = []
    
    # Validate age
    if not isinstance(age, (int, float)) or age < 0 or age > 150:
        errors.append(f"Invalid age: {age}. Must be between 0 and 150.")
    
    # Validate gender
    if gender.upper() not in ['M', 'F']:
        errors.append(f"Invalid gender: {gender}. Must be 'M' or 'F'.")
    
    # Validate view
    if view.upper() not in ['AP', 'PA']:
        errors.append(f"Invalid view: {view}. Must be 'AP' or 'PA'.")
    
    # Validate image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.format not in ['PNG', 'JPEG', 'JPG']:
            errors.append(f"Invalid image format: {image.format}. Must be PNG or JPEG.")
    except Exception as e:
        errors.append(f"Could not read image: {str(e)}")
    
    if errors:
        return None, None, errors
    
    # Preprocess inputs
    image_batch = preprocess_image_bytes(image_bytes)
    metadata_batch = preprocess_metadata(int(age), gender, view)
    
    return image_batch, metadata_batch, None


def get_original_image_array(image_bytes: bytes) -> np.ndarray:
    """
    Get the original image as numpy array (for GRAD-CAM overlay).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Original image resized to model input size, shape (256, 256, 3)
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.array(image)
