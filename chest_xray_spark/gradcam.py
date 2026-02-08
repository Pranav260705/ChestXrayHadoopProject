"""
GRAD-CAM Implementation for Model Interpretability

Generates class activation heatmaps to visualize which regions of the
chest X-ray the model focuses on for each prediction.

Based on: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
"""
import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple, Optional
import logging

from .config import IMAGE_SIZE, LABELS
from .model_loader import get_last_conv_layer_name

logger = logging.getLogger(__name__)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Find the name of the last Conv2D layer in the model.
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Name of the last Conv2D layer
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Check nested models (e.g., DenseNet backbone)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return f"{layer.name}/{sublayer.name}"
    
    raise ValueError("No Conv2D layer found in model")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    metadata_batch: np.ndarray,
    class_index: int
) -> np.ndarray:
    """
    Generate GRAD-CAM heatmap for a specific class.
    
    Args:
        model: Loaded Keras model
        image_batch: Preprocessed image of shape (1, 256, 256, 3)
        metadata_batch: Metadata of shape (1, 4)
        class_index: Index of the class to generate heatmap for
        
    Returns:
        Heatmap array of shape (256, 256) with values in [0, 1]
    """
    # Find the last convolutional layer
    last_conv_layer_name = find_last_conv_layer(model)
    logger.debug(f"Using layer '{last_conv_layer_name}' for GRAD-CAM")
    
    # Create a model that outputs both the conv layer output and predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        # Handle nested layer names
        logger.warning(f"Could not get layer directly, trying alternative: {e}")
        # Fallback: iterate through layers to find conv output
        conv_output = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_output = layer.output
        if conv_output is None:
            raise ValueError("Could not find Conv2D layer output")
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_output, model.output]
        )
    
    # Convert inputs to tensors
    img_tensor = tf.convert_to_tensor(image_batch)
    meta_tensor = tf.convert_to_tensor(metadata_batch)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model([img_tensor, meta_tensor])
        class_output = predictions[:, class_index]
    
    # Get gradients of the class output with respect to conv layer outputs
    grads = tape.gradient(class_output, conv_outputs)
    
    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # Apply ReLU to focus on positive contributions
    heatmap = tf.nn.relu(heatmap)
    
    # Normalize heatmap
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    # Convert to numpy and resize to image size
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    
    return heatmap


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay GRAD-CAM heatmap on original image.
    
    Args:
        original_image: Original RGB image of shape (256, 256, 3), values in [0, 255]
        heatmap: Heatmap of shape (256, 256), values in [0, 1]
        alpha: Transparency of the heatmap overlay
        
    Returns:
        Overlay image of shape (256, 256, 3), values in [0, 255]
    """
    # Convert heatmap to colormap (JET)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Ensure original image is in correct format
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Resize if needed
    if original_image.shape[:2] != IMAGE_SIZE:
        original_image = cv2.resize(original_image, IMAGE_SIZE)
    
    # Create overlay
    overlay = cv2.addWeighted(
        original_image.astype(np.float32), 1 - alpha,
        heatmap_color.astype(np.float32), alpha,
        0
    )
    
    return overlay.astype(np.uint8)


def generate_all_heatmaps(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    metadata_batch: np.ndarray,
    original_image: np.ndarray,
    top_k: int = 3,
    alpha: float = 0.4
) -> Tuple[dict, dict]:
    """
    Generate GRAD-CAM heatmaps for top K predictions.
    
    Args:
        model: Loaded Keras model
        image_batch: Preprocessed image
        metadata_batch: Metadata array
        original_image: Original image for overlay
        top_k: Number of top predictions
        alpha: Overlay transparency
        
    Returns:
        Tuple of (predictions_dict, overlays_dict)
    """
    # Get predictions
    predictions = model.predict([image_batch, metadata_batch], verbose=0)[0]
    pred_dict = {label: float(prob) for label, prob in zip(LABELS, predictions)}
    
    # Get top K classes
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    # Generate heatmaps
    overlays = {}
    for idx in top_indices:
        label = LABELS[idx]
        prob = predictions[idx]
        
        heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, idx)
        overlay = overlay_gradcam(original_image, heatmap, alpha)
        
        overlays[label] = {
            'heatmap': heatmap,
            'overlay': overlay,
            'probability': float(prob)
        }
    
    return pred_dict, overlays


def save_gradcam_visualization(
    overlay: np.ndarray,
    output_path: str,
    label: str,
    probability: float
) -> None:
    """
    Save GRAD-CAM visualization with label annotation.
    
    Args:
        overlay: Overlay image
        output_path: Path to save the image
        label: Disease label
        probability: Prediction probability
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert to PIL Image
    img = Image.fromarray(overlay)
    
    # Add text annotation
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    text = f"{label}: {probability*100:.1f}%"
    
    # Draw text with background
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw semi-transparent background
    draw.rectangle([5, 5, text_width + 15, text_height + 15], fill=(0, 0, 0, 180))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    
    # Save
    img.save(output_path)


def create_comparison_grid(
    original: np.ndarray,
    overlays: dict,
    output_path: str = None
) -> np.ndarray:
    """
    Create a side-by-side comparison grid of original and heatmaps.
    
    Args:
        original: Original image
        overlays: Dictionary of label -> overlay info
        output_path: Optional path to save the grid
        
    Returns:
        Grid image as numpy array
    """
    from PIL import Image
    
    # Ensure original is in correct format
    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)
    
    images = [original]
    
    for label, info in overlays.items():
        images.append(info['overlay'])
    
    # Create grid
    n_images = len(images)
    grid_width = IMAGE_SIZE[0] * n_images
    grid_height = IMAGE_SIZE[1]
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for i, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        grid.paste(img, (i * IMAGE_SIZE[0], 0))
    
    grid_array = np.array(grid)
    
    if output_path:
        grid.save(output_path)
    
    return grid_array
