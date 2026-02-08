"""
Flask Web Application for Chest X-Ray Inference

Provides a web interface for:
- Uploading chest X-ray images
- Entering patient metadata (age, gender, view)
- Running inference
- Displaying predictions with GRAD-CAM heatmaps
"""
import os
import io
import json
import base64
import logging
from datetime import datetime
from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify, 
    send_file,
    url_for
)
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from chest_xray_spark.config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    LABELS
)
from chest_xray_spark.model_loader import load_model_local, get_model_info
from chest_xray_spark.preprocessing import (
    preprocess_image_bytes,
    preprocess_metadata,
    get_original_image_array,
    validate_input
)
from chest_xray_spark.inference import (
    run_single_inference,
    get_top_predictions,
    format_predictions
)
from chest_xray_spark.gradcam import (
    generate_gradcam_heatmap,
    overlay_gradcam,
    generate_all_heatmaps
)
from chest_xray_spark.hdfs_utils import (
    save_inference_results,
    list_all_patients,
    get_patient_data,
    read_bytes_from_hdfs
)
from chest_xray_spark.batch_processor import run_spark_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model (loaded once on startup)
_model = None


def get_model():
    """Get or load the model (singleton pattern)."""
    global _model
    if _model is None:
        logger.info("Loading model...")
        _model = load_model_local()
        logger.info("Model loaded successfully")
    return _model


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Convert numpy array to base64 string for HTML embedding."""
    from PIL import Image
    
    # Ensure correct format
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype('uint8')
    
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html', labels=LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request.
    
    Expects:
        - file: Image file (PNG/JPEG)
        - age: Patient age (integer)
        - gender: 'M' or 'F'
        - view: 'AP' or 'PA'
        
    Returns:
        JSON with predictions and GRAD-CAM images
    """
    try:
        # Validate file presence
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG or JPEG.'}), 400
        
        # Get metadata
        patient_name = request.form.get('patient_name', '').strip()
        if not patient_name:
            return jsonify({'error': 'Patient name is required'}), 400
        
        try:
            age = int(request.form.get('age', 50))
        except ValueError:
            return jsonify({'error': 'Invalid age value'}), 400
            
        gender = request.form.get('gender', 'M').upper()
        view = request.form.get('view', 'PA').upper()
        
        # Sanitize patient name for folder (remove spaces and special chars)
        import re
        sanitized_name = re.sub(r'[^a-zA-Z]', '', patient_name).title()
        if not sanitized_name:
            sanitized_name = "Unknown"
        
        # Read image bytes
        image_bytes = file.read()
        
        # Validate inputs
        image_batch, metadata_batch, errors = validate_input(
            image_bytes, age, gender, view
        )
        
        if errors:
            return jsonify({'error': '\n'.join(errors)}), 400
        
        # Load model and run inference
        model = get_model()
        predictions = run_single_inference(model, image_batch, metadata_batch)
        
        # Get top predictions
        top_preds = get_top_predictions(predictions, top_k=5)
        
        # Get original image for GRAD-CAM overlay
        original_image = get_original_image_array(image_bytes)
        
        # Save results to HDFS/local
        # Folder format: PatientName + YYYYMMDD (e.g., JohnDoe20260209)
        date_str = datetime.now().strftime("%Y%m%d")
        folder_name = f"{sanitized_name}{date_str}"
        
        metadata = {
            'patient_name': patient_name,
            'age': age,
            'gender': gender,
            'view': view,
            'timestamp': datetime.now().isoformat(),
            'folder_name': folder_name,
            'source': 'web_upload'
        }
        
        # Save to HDFS (or local filesystem based on USE_LOCAL_FS env)
        try:
            # For GRAD-CAM images, we need to convert from base64 to arrays for saving
            gradcam_arrays = {}
            for label, prob in top_preds[:3]:
                class_idx = LABELS.index(label) 
                heatmap = generate_gradcam_heatmap(model, image_batch, metadata_batch, class_idx)
                overlay = overlay_gradcam(original_image, heatmap)
                gradcam_arrays[label] = overlay
            
            result_dir = save_inference_results(
                folder_name=folder_name,
                original_image=image_bytes,
                predictions=predictions,
                metadata=metadata,
                gradcam_images=gradcam_arrays
            )
            logger.info(f"Results saved to: {result_dir}")
        except Exception as save_error:
            logger.warning(f"Could not save to HDFS: {save_error}")
            result_dir = None
        
        # Convert GRAD-CAM arrays to base64 for response
        gradcam_images = {label: image_to_base64(arr) for label, arr in gradcam_arrays.items()}
        
        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'top_predictions': [
                {'label': label, 'probability': prob}
                for label, prob in top_preds
            ],
            'gradcam_images': gradcam_images,
            'original_image': image_to_base64(original_image),
            'metadata': metadata,
            'folder_name': folder_name,
            'result_dir': result_dir
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    """Return model information."""
    try:
        model = get_model()
        info = get_model_info(model)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/labels')
def get_labels():
    """Return list of disease labels."""
    return jsonify({'labels': LABELS})


@app.route('/search')
def search_page():
    """Render the patient search page."""
    return render_template('search.html')


@app.route('/api/patients')
def api_list_patients():
    """API endpoint to list all patients."""
    try:
        patients = list_all_patients()
        return jsonify({'patients': patients})
    except Exception as e:
        logger.error(f"Error listing patients: {e}")
        return jsonify({'patients': [], 'error': str(e)})


@app.route('/api/patients/<folder_name>')
def api_get_patient(folder_name):
    """API endpoint to get patient details."""
    try:
        data = get_patient_data(folder_name)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting patient data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/images/<folder_name>/<filename>')
def serve_image(folder_name, filename):
    """Serve images from HDFS/local storage."""
    try:
        hdfs_path = f"/ChestXRayDB/{folder_name}/{filename}"
        image_bytes = read_bytes_from_hdfs(hdfs_path)
        if image_bytes:
            return send_file(
                io.BytesIO(image_bytes),
                mimetype='image/png'
            )
        else:
            return 'Image not found', 404
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return 'Error loading image', 500


@app.route('/batch')
def batch_page():
    """Render the batch processing page."""
    return render_template('batch.html')


@app.route('/batch/process', methods=['POST'])
def batch_process():
    """Handle batch upload and process with Spark."""
    import json
    
    try:
        patient_count = int(request.form.get('patient_count', 0))
        
        if patient_count == 0:
            return jsonify({'error': 'No patients provided'}), 400
        
        # Collect patient data
        patients = []
        for i in range(patient_count):
            patient_json = request.form.get(f'patient_{i}')
            image_file = request.files.get(f'image_{i}')
            
            if patient_json and image_file:
                patient_data = json.loads(patient_json)
                patient_data['image_bytes'] = image_file.read()
                patients.append(patient_data)
        
        if not patients:
            return jsonify({'error': 'No valid patients found'}), 400
        
        logger.info(f"Processing batch of {len(patients)} patients with Spark")
        
        # Run Spark batch processing
        results = run_spark_batch(patients)
        
        return jsonify({
            'success': True,
            'processed': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': _model is not None,
        'timestamp': datetime.now().isoformat()
    })


def main():
    """Run the Flask application."""
    logger.info("=" * 60)
    logger.info("Chest X-Ray Inference Web Application")
    logger.info("=" * 60)
    logger.info(f"Starting server on http://{FLASK_HOST}:{FLASK_PORT}")
    
    # Pre-load model
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        logger.info("Model will be loaded on first request")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )


if __name__ == '__main__':
    main()
