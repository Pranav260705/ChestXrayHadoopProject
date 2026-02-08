"""
TensorFlow Model Loading Utilities

Supports loading the multimodal DenseNet121 model from:
- Local filesystem (SavedModel format)
- HDFS (when running in Spark cluster)
"""
import os
import logging
import tempfile
import shutil
import tensorflow as tf

from .config import (
    LOCAL_MODEL_PATH, 
    LOCAL_MODEL_PB, 
    LOCAL_VARIABLES_DIR,
    HDFS_MODEL_PATH,
    LABELS,
    META_COLS
)

logger = logging.getLogger(__name__)

# Global model cache for efficiency
_MODEL_CACHE = {}


def load_model_local(model_path: str = None):
    """
    Load TensorFlow SavedModel from local filesystem.
    
    The model structure from the notebook:
    - saved_model.pb (in base directory)
    - Variables/ directory with variables.index and variables.data-*
    
    Args:
        model_path: Path to SavedModel directory. If None, uses default path.
        
    Returns:
        Loaded Keras model or InferenceWrapper
    """
    if model_path is None:
        # Check if we have SavedModel format directly
        if os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
        else:
            # Reconstruct SavedModel from pb and Variables
            model_path = _reconstruct_savedmodel()
    
    logger.info(f"Loading model from: {model_path}")
    
    # Check cache first
    if model_path in _MODEL_CACHE:
        logger.info("Returning cached model")
        return _MODEL_CACHE[model_path]
    
    # Try Keras load_model first (works with TF 2.15 + Keras 2.x)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        _MODEL_CACHE[model_path] = model
        logger.info(f"Model loaded with Keras. Input shapes: {[inp.shape for inp in model.inputs]}")
        return model
    except Exception as e:
        logger.warning(f"Keras load_model failed: {e}, trying signature-based loading...")
    
    # Fallback: use SavedModel signatures
    try:
        loaded = tf.saved_model.load(model_path)
        
        infer_fn = None
        if hasattr(loaded, 'signatures'):
            infer_fn = loaded.signatures.get('serving_default')
            if infer_fn is None:
                sig_keys = list(loaded.signatures.keys())
                if sig_keys:
                    infer_fn = loaded.signatures[sig_keys[0]]
        
        if infer_fn:
            model = InferenceWrapper(infer_fn)
            _MODEL_CACHE[model_path] = model
            logger.info("Model loaded using SavedModel signatures")
            return model
        
        raise ValueError("No usable inference function found in SavedModel")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


class InferenceWrapper:
    """Wrapper for SavedModel inference function to provide Keras-like predict interface."""
    
    def __init__(self, infer_fn):
        self.infer_fn = infer_fn
        self.input_keys = []
        self.output_keys = []
        
        # Extract input/output key names from signature
        if hasattr(infer_fn, 'structured_input_signature'):
            sig = infer_fn.structured_input_signature
            if sig and len(sig) >= 2 and isinstance(sig[1], dict):
                self.input_keys = list(sig[1].keys())
        
        if hasattr(infer_fn, 'structured_outputs'):
            self.output_keys = list(infer_fn.structured_outputs.keys())
    
    def predict(self, inputs, verbose=0):
        """Run prediction matching Keras model.predict interface."""
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            img, meta = inputs
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            meta_tensor = tf.convert_to_tensor(meta, dtype=tf.float32)
            
            # Try with extracted input keys
            if len(self.input_keys) >= 2:
                result = self.infer_fn(**{
                    self.input_keys[0]: img_tensor,
                    self.input_keys[1]: meta_tensor
                })
            else:
                # Try common fallback names
                for keys in [('input_1', 'input_2'), ('args_0', 'args_1')]:
                    try:
                        result = self.infer_fn(**{keys[0]: img_tensor, keys[1]: meta_tensor})
                        break
                    except TypeError:
                        continue
                else:
                    raise ValueError("Could not determine input key names")
            
            # Extract output
            if isinstance(result, dict) and self.output_keys:
                return result[self.output_keys[0]].numpy()
            elif isinstance(result, dict):
                return list(result.values())[0].numpy()
            return result.numpy() if hasattr(result, 'numpy') else result
        
        raise ValueError("Expected inputs as [image_batch, metadata_batch]")
    
    @property
    def inputs(self):
        """Mock inputs property for compatibility."""
        class MockInput:
            def __init__(self, shape):
                self.shape = shape
        return [MockInput((None, 256, 256, 3)), MockInput((None, 4))]




def _reconstruct_savedmodel() -> str:
    """
    Reconstruct SavedModel directory from separate pb and variables files.
    
    The user's model structure:
    - saved_model.pb (in base dir)
    - Variables/variables.index
    - Variables/variables.data-00000-of-00001
    
    Returns:
        Path to reconstructed SavedModel directory
    """
    # Create temporary directory with proper structure
    temp_dir = tempfile.mkdtemp(prefix="chest_xray_model_")
    
    try:
        # Copy saved_model.pb
        if os.path.exists(LOCAL_MODEL_PB):
            shutil.copy(LOCAL_MODEL_PB, os.path.join(temp_dir, "saved_model.pb"))
        else:
            raise FileNotFoundError(f"saved_model.pb not found at {LOCAL_MODEL_PB}")
        
        # Create variables directory and copy files
        variables_dir = os.path.join(temp_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        
        if os.path.exists(LOCAL_VARIABLES_DIR):
            for fname in os.listdir(LOCAL_VARIABLES_DIR):
                src = os.path.join(LOCAL_VARIABLES_DIR, fname)
                # Map filename to expected format
                if "index" in fname:
                    dst = os.path.join(variables_dir, "variables.index")
                elif "data" in fname:
                    dst = os.path.join(variables_dir, "variables.data-00000-of-00001")
                else:
                    continue
                shutil.copy(src, dst)
        else:
            raise FileNotFoundError(f"Variables directory not found at {LOCAL_VARIABLES_DIR}")
        
        logger.info(f"Reconstructed SavedModel at: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_model_from_hdfs(hdfs_path: str = None) -> tf.keras.Model:
    """
    Load model from HDFS by copying to local temp directory first.
    
    Args:
        hdfs_path: HDFS path to SavedModel directory
        
    Returns:
        Loaded Keras model
    """
    from pyspark import SparkFiles
    
    if hdfs_path is None:
        hdfs_path = HDFS_MODEL_PATH
    
    # Check cache first
    if hdfs_path in _MODEL_CACHE:
        return _MODEL_CACHE[hdfs_path]
    
    # Create temp directory for model files
    temp_dir = tempfile.mkdtemp(prefix="hdfs_model_")
    
    try:
        # Get SparkContext to access HDFS
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        
        # Copy model files from HDFS
        hadoop_conf = sc._jsc.hadoopConfiguration()
        
        from py4j.java_gateway import java_import
        java_import(sc._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
        java_import(sc._gateway.jvm, "org.apache.hadoop.fs.Path")
        
        fs = sc._gateway.jvm.FileSystem.get(hadoop_conf)
        
        # Copy saved_model.pb
        fs.copyToLocalFile(
            sc._gateway.jvm.Path(f"{hdfs_path}/saved_model.pb"),
            sc._gateway.jvm.Path(os.path.join(temp_dir, "saved_model.pb"))
        )
        
        # Create variables directory and copy files
        variables_dir = os.path.join(temp_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        
        fs.copyToLocalFile(
            sc._gateway.jvm.Path(f"{hdfs_path}/variables/variables.index"),
            sc._gateway.jvm.Path(os.path.join(variables_dir, "variables.index"))
        )
        
        fs.copyToLocalFile(
            sc._gateway.jvm.Path(f"{hdfs_path}/variables/variables.data-00000-of-00001"),
            sc._gateway.jvm.Path(os.path.join(variables_dir, "variables.data-00000-of-00001"))
        )
        
        # Load model
        model = tf.keras.models.load_model(temp_dir)
        _MODEL_CACHE[hdfs_path] = model
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from HDFS: {e}")
        raise
    finally:
        # Don't clean up temp_dir as TensorFlow may still need it
        pass


def get_model_info(model: tf.keras.Model) -> dict:
    """
    Get model architecture information.
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Dictionary with model info
    """
    return {
        "name": model.name,
        "input_shapes": [str(inp.shape) for inp in model.inputs],
        "output_shape": str(model.output.shape),
        "num_layers": len(model.layers),
        "trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "labels": LABELS,
        "meta_cols": META_COLS
    }


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Find the name of the last Conv2D layer (for GRAD-CAM).
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Name of the last Conv2D layer
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Also check for nested layers (in case of functional API)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    
    raise ValueError("No Conv2D layer found in model")
