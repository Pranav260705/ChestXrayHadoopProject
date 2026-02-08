"""
Spark Job Entry Point for Chest X-Ray Inference

This module provides the main entry point for running distributed
inference on Apache Spark.

Usage:
    spark-submit chest_xray_spark/spark_job.py --input <path> --output <path>
"""
import argparse
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "ChestXRayInference", local: bool = False):
    """
    Create and configure Spark session.
    
    Args:
        app_name: Application name
        local: If True, run in local mode
        
    Returns:
        SparkSession
    """
    from pyspark.sql import SparkSession
    
    builder = SparkSession.builder.appName(app_name)
    
    if local:
        builder = builder.master("local[*]")
    
    # Configure for TensorFlow
    builder = builder.config("spark.executor.memory", "8g")
    builder = builder.config("spark.driver.memory", "4g")
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def process_batch(
    spark,
    input_path: str,
    output_path: str,
    model_path: str = None
):
    """
    Process a batch of images from HDFS.
    
    Args:
        spark: SparkSession
        input_path: Path to input directory containing images and metadata
        output_path: Path to output directory for results
        model_path: Path to TensorFlow model
    """
    from pyspark.sql.functions import col, udf, lit
    from pyspark.sql.types import MapType, StringType, FloatType, StructType, StructField
    import json
    
    from .model_loader import load_model_local
    from .preprocessing import preprocess_image_bytes, preprocess_metadata
    from .inference import run_single_inference
    from .gradcam import generate_gradcam_heatmap, overlay_gradcam
    from .config import LABELS
    
    logger.info(f"Processing batch from: {input_path}")
    
    # Read input data
    # Expected format: directory with images and a metadata.json file
    # or a CSV/Parquet file with image paths and metadata
    
    # Try reading as structured data first
    try:
        df = spark.read.parquet(f"{input_path}/data.parquet")
    except:
        try:
            df = spark.read.csv(f"{input_path}/data.csv", header=True, inferSchema=True)
        except:
            # Assume directory of images - create DataFrame from file listing
            from pyspark.sql.types import StructType, StructField, StringType
            
            # List files in directory
            files = spark.sparkContext.wholeTextFiles(f"{input_path}/*.png").keys().collect()
            
            schema = StructType([
                StructField("image_path", StringType(), False)
            ])
            
            df = spark.createDataFrame([(f,) for f in files], schema)
            
            # Add default metadata columns if not present
            df = df.withColumn("age_norm", lit(0.5))
            df = df.withColumn("sex_bin", lit(0.0))
            df = df.withColumn("view_AP", lit(0.0))
            df = df.withColumn("view_PA", lit(1.0))
    
    logger.info(f"Loaded {df.count()} records")
    
    # Load model on driver (will be broadcast to executors)
    model = load_model_local(model_path)
    
    # Broadcast model weights (for large models, consider saving to shared storage)
    # For now, each executor will load the model independently
    
    # Define UDF for inference
    @udf(MapType(StringType(), FloatType()))
    def inference_udf(image_path, age_norm, sex_bin, view_ap, view_pa):
        import numpy as np
        
        try:
            # Load model (cached per executor)
            model = load_model_local(model_path)
            
            # Read image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            image_batch = preprocess_image_bytes(image_bytes)
            metadata_batch = np.array([[age_norm, sex_bin, view_ap, view_pa]], dtype=np.float32)
            
            # Run inference
            preds = model.predict([image_batch, metadata_batch], verbose=0)[0]
            return {label: float(prob) for label, prob in zip(LABELS, preds)}
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {label: 0.0 for label in LABELS}
    
    # Run inference
    result_df = df.withColumn(
        "predictions",
        inference_udf(
            col("image_path"),
            col("age_norm"),
            col("sex_bin"),
            col("view_AP"),
            col("view_PA")
        )
    )
    
    # Save results
    result_df.write.mode("overwrite").parquet(f"{output_path}/predictions")
    
    logger.info(f"Results saved to: {output_path}/predictions")
    
    return result_df


def run_single_job(
    image_path: str,
    age: int,
    gender: str,
    view: str,
    output_dir: str,
    model_path: str = None,
    generate_heatmaps: bool = True
):
    """
    Run inference on a single image (for testing or Flask integration).
    
    Args:
        image_path: Path to input image
        age: Patient age
        gender: Patient gender
        view: View position
        output_dir: Output directory
        model_path: Path to model
        generate_heatmaps: Whether to generate GRAD-CAM heatmaps
        
    Returns:
        Dictionary with predictions and file paths
    """
    import numpy as np
    from datetime import datetime
    
    from .model_loader import load_model_local
    from .preprocessing import preprocess_image_bytes, preprocess_metadata, get_original_image_array
    from .inference import run_single_inference, get_top_predictions
    from .gradcam import generate_gradcam_heatmap, overlay_gradcam
    from .hdfs_utils import save_inference_results
    from .config import LABELS
    
    logger.info(f"Processing single image: {image_path}")
    
    # Load model
    model = load_model_local(model_path)
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Preprocess
    image_batch = preprocess_image_bytes(image_bytes)
    metadata_batch = preprocess_metadata(age, gender, view)
    
    # Run inference
    predictions = run_single_inference(model, image_batch, metadata_batch)
    
    # Generate GRAD-CAM heatmaps for top predictions
    gradcam_overlays = {}
    
    if generate_heatmaps:
        original_image = get_original_image_array(image_bytes)
        top_preds = get_top_predictions(predictions, top_k=3)
        
        for label, prob in top_preds:
            class_idx = LABELS.index(label)
            heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, class_idx)
            overlay = overlay_gradcam(original_image, heatmap)
            gradcam_overlays[label] = overlay
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_id = os.path.basename(image_path).split('.')[0]
    
    metadata = {
        "age": age,
        "gender": gender,
        "view": view,
        "age_norm": float(metadata_batch[0][0]),
        "sex_bin": float(metadata_batch[0][1]),
        "view_AP": float(metadata_batch[0][2]),
        "view_PA": float(metadata_batch[0][3]),
        "timestamp": timestamp
    }
    
    result_dir = save_inference_results(
        patient_id=patient_id,
        timestamp=timestamp,
        original_image=image_bytes,
        predictions=predictions,
        metadata=metadata,
        gradcam_images=gradcam_overlays
    )
    
    return {
        "predictions": predictions,
        "result_dir": result_dir,
        "top_predictions": get_top_predictions(predictions, 3)
    }


def main():
    """Main entry point for Spark job."""
    parser = argparse.ArgumentParser(description="Chest X-Ray Inference Spark Job")
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input path (image file or directory)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to TensorFlow model"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (no cluster)"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process single image (not batch)"
    )
    
    parser.add_argument(
        "--age",
        type=int,
        default=50,
        help="Patient age (for single mode)"
    )
    
    parser.add_argument(
        "--gender",
        type=str,
        default="M",
        choices=["M", "F"],
        help="Patient gender (for single mode)"
    )
    
    parser.add_argument(
        "--view",
        type=str,
        default="PA",
        choices=["AP", "PA"],
        help="View position (for single mode)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Chest X-Ray Inference Spark Job")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {'Local' if args.local else 'Cluster'}")
    
    if args.single:
        # Single image processing (no Spark)
        result = run_single_job(
            image_path=args.input,
            age=args.age,
            gender=args.gender,
            view=args.view,
            output_dir=args.output,
            model_path=args.model
        )
        
        logger.info("Results:")
        for label, prob in result["top_predictions"]:
            logger.info(f"  {label}: {prob*100:.2f}%")
        logger.info(f"Output saved to: {result['result_dir']}")
        
    else:
        # Batch processing with Spark
        spark = create_spark_session(local=args.local)
        
        try:
            process_batch(
                spark=spark,
                input_path=args.input,
                output_path=args.output,
                model_path=args.model
            )
        finally:
            spark.stop()
    
    logger.info("Job completed successfully")


if __name__ == "__main__":
    main()
