"""
Spark Batch Processor for Chest X-Ray Inference

Processes multiple patient images in parallel using Apache Spark.
"""
import os
import re
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """Sanitize patient name for folder naming."""
    clean = re.sub(r'[^a-zA-Z]', '', name).title()
    return clean if clean else "Unknown"


def run_spark_batch(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple patients using Spark for distributed inference.
    
    Args:
        patients: List of dicts with keys:
            - name: Patient name
            - age: Patient age
            - gender: 'M' or 'F'
            - view: 'AP' or 'PA'
            - image_bytes: Image file bytes
            
    Returns:
        List of result dicts with success status and folder names
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType
    
    logger.info(f"Starting Spark batch processing for {len(patients)} patients")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ChestXRayBatchInference") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        results = []
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Process each patient - Spark parallelizes across executors
        # For web upload scenario, we use Spark's parallel processing capability
        
        # Create RDD from patient data for parallel processing
        patient_data = []
        for i, patient in enumerate(patients):
            patient_data.append({
                'index': i,
                'name': patient['name'],
                'age': patient['age'],
                'gender': patient['gender'],
                'view': patient['view'],
                'image_bytes': patient['image_bytes'],
                'folder_name': f"{sanitize_name(patient['name'])}{date_str}"
            })
        
        # Create RDD and process in parallel
        patient_rdd = spark.sparkContext.parallelize(patient_data, numSlices=min(len(patient_data), 4))
        
        # Process each patient using Spark workers
        def process_patient(patient_dict):
            """Process a single patient - runs on Spark executor."""
            try:
                import numpy as np
                from chest_xray_spark.model_loader import load_model_local
                from chest_xray_spark.preprocessing import preprocess_image_bytes, preprocess_metadata, get_original_image_array
                from chest_xray_spark.inference import run_single_inference, get_top_predictions
                from chest_xray_spark.gradcam import generate_gradcam_heatmap, overlay_gradcam
                from chest_xray_spark.hdfs_utils import save_inference_results
                from chest_xray_spark.config import LABELS
                
                # Load model
                model = load_model_local()
                
                # Preprocess
                image_bytes = patient_dict['image_bytes']
                image_batch = preprocess_image_bytes(image_bytes)
                metadata_batch = preprocess_metadata(
                    patient_dict['age'],
                    patient_dict['gender'],
                    patient_dict['view']
                )
                
                # Run inference
                predictions = run_single_inference(model, image_batch, metadata_batch)
                top_preds = get_top_predictions(predictions, top_k=3)
                
                # Generate GRAD-CAM heatmaps
                original_image = get_original_image_array(image_bytes)
                gradcam_arrays = {}
                for label, prob in top_preds:
                    class_idx = LABELS.index(label)
                    heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, class_idx)
                    overlay = overlay_gradcam(original_image, heatmap)
                    gradcam_arrays[label] = overlay
                
                # Save results
                metadata = {
                    'patient_name': patient_dict['name'],
                    'age': patient_dict['age'],
                    'gender': patient_dict['gender'],
                    'view': patient_dict['view'],
                    'timestamp': datetime.now().isoformat(),
                    'folder_name': patient_dict['folder_name'],
                    'source': 'spark_batch'
                }
                
                result_dir = save_inference_results(
                    folder_name=patient_dict['folder_name'],
                    original_image=image_bytes,
                    predictions=predictions,
                    metadata=metadata,
                    gradcam_images=gradcam_arrays
                )
                
                return {
                    'index': patient_dict['index'],
                    'patient_name': patient_dict['name'],
                    'folder_name': patient_dict['folder_name'],
                    'success': True,
                    'message': f"Saved to {result_dir}",
                    'top_prediction': top_preds[0][0] if top_preds else None
                }
                
            except Exception as e:
                return {
                    'index': patient_dict['index'],
                    'patient_name': patient_dict['name'],
                    'folder_name': patient_dict.get('folder_name', 'unknown'),
                    'success': False,
                    'message': str(e)
                }
        
        # Run parallel processing with Spark
        logger.info("Distributing inference across Spark executors...")
        results_rdd = patient_rdd.map(process_patient)
        
        # Collect results
        results = results_rdd.collect()
        
        # Sort by original index
        results.sort(key=lambda x: x['index'])
        
        logger.info(f"Batch processing complete. {sum(1 for r in results if r['success'])}/{len(results)} successful")
        
        return results
        
    finally:
        spark.stop()
        logger.info("Spark session stopped")


def run_batch_sequential(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback: Process patients sequentially without Spark.
    Used when Spark is not available.
    """
    from .model_loader import load_model_local
    from .preprocessing import preprocess_image_bytes, preprocess_metadata, get_original_image_array
    from .inference import run_single_inference, get_top_predictions
    from .gradcam import generate_gradcam_heatmap, overlay_gradcam
    from .hdfs_utils import save_inference_results
    from .config import LABELS
    
    logger.info(f"Processing {len(patients)} patients sequentially (Spark fallback)")
    
    model = load_model_local()
    results = []
    date_str = datetime.now().strftime("%Y%m%d")
    
    for i, patient in enumerate(patients):
        try:
            folder_name = f"{sanitize_name(patient['name'])}{date_str}"
            image_bytes = patient['image_bytes']
            
            # Preprocess
            image_batch = preprocess_image_bytes(image_bytes)
            metadata_batch = preprocess_metadata(
                patient['age'],
                patient['gender'],
                patient['view']
            )
            
            # Inference
            predictions = run_single_inference(model, image_batch, metadata_batch)
            top_preds = get_top_predictions(predictions, top_k=3)
            
            # GRAD-CAM
            original_image = get_original_image_array(image_bytes)
            gradcam_arrays = {}
            for label, prob in top_preds:
                class_idx = LABELS.index(label)
                heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, class_idx)
                overlay = overlay_gradcam(original_image, heatmap)
                gradcam_arrays[label] = overlay
            
            # Save
            metadata = {
                'patient_name': patient['name'],
                'age': patient['age'],
                'gender': patient['gender'],
                'view': patient['view'],
                'timestamp': datetime.now().isoformat(),
                'folder_name': folder_name,
                'source': 'batch_sequential'
            }
            
            result_dir = save_inference_results(
                folder_name=folder_name,
                original_image=image_bytes,
                predictions=predictions,
                metadata=metadata,
                gradcam_images=gradcam_arrays
            )
            
            results.append({
                'index': i,
                'patient_name': patient['name'],
                'folder_name': folder_name,
                'success': True,
                'message': f"Saved to {result_dir}"
            })
            
        except Exception as e:
            results.append({
                'index': i,
                'patient_name': patient['name'],
                'folder_name': folder_name if 'folder_name' in locals() else 'unknown',
                'success': False,
                'message': str(e)
            })
    
    return results
