"""
Inference Module for Chest X-Ray Classification

Provides both single-sample and Spark batch inference capabilities.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging

from .config import LABELS, META_COLS
from .model_loader import load_model_local, get_last_conv_layer_name
from .preprocessing import (
    preprocess_image_bytes, 
    preprocess_metadata,
    preprocess_image_tf
)

logger = logging.getLogger(__name__)


def run_single_inference(
    model,
    image_batch: np.ndarray,
    metadata_batch: np.ndarray
) -> Dict[str, float]:
    """
    Run inference on a single sample.
    
    Args:
        model: Loaded Keras model
        image_batch: Preprocessed image array of shape (1, 256, 256, 3)
        metadata_batch: Metadata array of shape (1, 4)
        
    Returns:
        Dictionary mapping disease labels to probabilities
    """
    # Run prediction
    predictions = model.predict([image_batch, metadata_batch], verbose=0)
    
    # Convert to dictionary
    probs = predictions[0]
    result = {label: float(prob) for label, prob in zip(LABELS, probs)}
    
    return result


def run_inference_from_bytes(
    model: tf.keras.Model,
    image_bytes: bytes,
    age: int,
    gender: str,
    view: str
) -> Dict[str, float]:
    """
    Run inference from raw image bytes and patient info.
    
    Args:
        model: Loaded Keras model
        image_bytes: Raw image bytes (PNG/JPEG)
        age: Patient age in years
        gender: 'M' or 'F'
        view: 'AP' or 'PA'
        
    Returns:
        Dictionary mapping disease labels to probabilities
    """
    # Preprocess inputs
    image_batch = preprocess_image_bytes(image_bytes)
    metadata_batch = preprocess_metadata(age, gender, view)
    
    return run_single_inference(model, image_batch, metadata_batch)


def get_top_predictions(
    predictions: Dict[str, float],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Get top K predictions sorted by probability.
    
    Args:
        predictions: Dictionary of label -> probability
        top_k: Number of top predictions to return
        
    Returns:
        List of (label, probability) tuples sorted by probability descending
    """
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:top_k]


def format_predictions(predictions: Dict[str, float]) -> str:
    """
    Format predictions as a human-readable string.
    
    Args:
        predictions: Dictionary of label -> probability
        
    Returns:
        Formatted string
    """
    lines = []
    for label, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 20)
        lines.append(f"{label:25s} {prob*100:6.2f}% {bar}")
    return "\n".join(lines)


# ==============================================================================
# Spark Batch Inference Functions
# ==============================================================================

def create_inference_udf(model_path: str = None):
    """
    Create a Spark pandas UDF for batch inference.
    
    Usage in Spark:
        from chest_xray_spark.inference import create_inference_udf
        inference_udf = create_inference_udf()
        df = df.withColumn("predictions", inference_udf(col("image_path"), col("metadata")))
    
    Args:
        model_path: Path to model (local or HDFS)
        
    Returns:
        Spark pandas UDF
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import MapType, StringType, FloatType
    import pandas as pd
    
    @pandas_udf(MapType(StringType(), FloatType()))
    def inference_udf(
        image_paths: pd.Series,
        age_norm: pd.Series,
        sex_bin: pd.Series,
        view_ap: pd.Series,
        view_pa: pd.Series
    ) -> pd.Series:
        """
        Pandas UDF for batch inference.
        """
        # Load model (cached per executor)
        model = load_model_local(model_path)
        
        results = []
        
        for path, age, sex, ap, pa in zip(
            image_paths, age_norm, sex_bin, view_ap, view_pa
        ):
            try:
                # Load and preprocess image
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                image_batch = preprocess_image_bytes(image_bytes)
                
                # Create metadata batch
                metadata_batch = np.array([[age, sex, ap, pa]], dtype=np.float32)
                
                # Run inference
                preds = model.predict([image_batch, metadata_batch], verbose=0)[0]
                result = {label: float(prob) for label, prob in zip(LABELS, preds)}
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({label: 0.0 for label in LABELS})
        
        return pd.Series(results)
    
    return inference_udf


def batch_inference_local(
    model: tf.keras.Model,
    image_paths: List[str],
    metadata_list: List[np.ndarray],
    batch_size: int = 32
) -> List[Dict[str, float]]:
    """
    Run batch inference on a list of images (local, non-Spark).
    
    Args:
        model: Loaded Keras model
        image_paths: List of paths to images
        metadata_list: List of metadata arrays, each of shape (4,)
        batch_size: Batch size for inference
        
    Returns:
        List of prediction dictionaries
    """
    all_predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_metadata = metadata_list[i:i + batch_size]
        
        # Load and preprocess images
        images = []
        for path in batch_paths:
            with open(path, 'rb') as f:
                img = preprocess_image_bytes(f.read())
            images.append(img[0])  # Remove batch dimension
        
        image_batch = np.stack(images, axis=0)
        metadata_batch = np.stack(batch_metadata, axis=0)
        
        # Run batch inference
        predictions = model.predict([image_batch, metadata_batch], verbose=0)
        
        # Convert to dictionaries
        for pred in predictions:
            result = {label: float(prob) for label, prob in zip(LABELS, pred)}
            all_predictions.append(result)
    
    return all_predictions


def predict_with_gradcam(
    model: tf.keras.Model,
    image_bytes: bytes,
    age: int,
    gender: str,
    view: str,
    top_k: int = 3
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Run inference and generate GRAD-CAM heatmaps for top predictions.
    
    Args:
        model: Loaded Keras model
        image_bytes: Raw image bytes
        age: Patient age
        gender: Patient gender
        view: View position
        top_k: Number of top predictions to generate heatmaps for
        
    Returns:
        Tuple of (predictions_dict, heatmaps_dict)
        heatmaps_dict maps disease label to heatmap array
    """
    from .gradcam import generate_gradcam_heatmap
    
    # Preprocess inputs
    image_batch = preprocess_image_bytes(image_bytes)
    metadata_batch = preprocess_metadata(age, gender, view)
    
    # Run inference
    predictions = run_single_inference(model, image_batch, metadata_batch)
    
    # Get top predictions
    top_preds = get_top_predictions(predictions, top_k)
    
    # Generate GRAD-CAM heatmaps
    heatmaps = {}
    for label, prob in top_preds:
        class_index = LABELS.index(label)
        heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, class_index)
        heatmaps[label] = heatmap
    
    return predictions, heatmaps
