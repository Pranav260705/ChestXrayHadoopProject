"""
Configuration settings for Chest X-Ray Inference System
"""
import os
import json

# ==============================================================================
# Model Configuration
# ==============================================================================

# Image preprocessing
IMAGE_SIZE = (256, 256)
IMAGE_CHANNELS = 3

# Model schema - from model_schema2.json
LABELS = [
    "Cardiomegaly", "Emphysema", "Effusion", "Hernia", "Infiltration",
    "Mass", "Nodule", "Atelectasis", "Pneumothorax", "Pleural_Thickening",
    "Pneumonia", "Fibrosis", "Edema", "Consolidation"
]

META_COLS = ["age_norm", "sex_bin", "view_AP", "view_PA"]

# ==============================================================================
# File Paths
# ==============================================================================

# Base directory (where the project is located)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local model path (TensorFlow SavedModel format)
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
LOCAL_MODEL_PB = os.path.join(BASE_DIR, "saved_model.pb")
LOCAL_VARIABLES_DIR = os.path.join(BASE_DIR, "Variables")

# ==============================================================================
# HDFS Configuration
# ==============================================================================

# HDFS namenode (change to your cluster's namenode)
HDFS_NAMENODE = os.environ.get("HDFS_NAMENODE", "hdfs://localhost:9000")

# HDFS paths
HDFS_BASE_PATH = "/chest_xray"
HDFS_MODEL_PATH = f"{HDFS_BASE_PATH}/model"
HDFS_UPLOADS_PATH = f"{HDFS_BASE_PATH}/uploads"
HDFS_RESULTS_PATH = f"{HDFS_BASE_PATH}/results"

# ==============================================================================
# Spark Configuration
# ==============================================================================

SPARK_APP_NAME = "ChestXRayInference"
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")

# Executor settings
SPARK_EXECUTOR_MEMORY = "8g"
SPARK_EXECUTOR_CORES = "4"
SPARK_DRIVER_MEMORY = "4g"

# ==============================================================================
# Flask Configuration
# ==============================================================================

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = True

# Upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Temporary upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# ==============================================================================
# Inference Configuration
# ==============================================================================

# GRAD-CAM settings
GRADCAM_ALPHA = 0.4  # Overlay transparency
TOP_K_PREDICTIONS = 3  # Number of top predictions to show heatmaps for

# Batch processing
BATCH_SIZE = 32

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_full_hdfs_path(path: str) -> str:
    """Get full HDFS path with namenode prefix"""
    return f"{HDFS_NAMENODE}{path}"

def load_model_schema(schema_path: str = None) -> dict:
    """Load model schema from JSON file"""
    if schema_path is None:
        schema_path = os.path.join(BASE_DIR, "model_schema2.json")
    
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return json.load(f)
    
    # Return default schema
    return {"labels": LABELS, "meta_cols": META_COLS}

def ensure_directories():
    """Create necessary local directories"""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create directories on import
ensure_directories()
