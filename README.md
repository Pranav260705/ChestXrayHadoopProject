# Chest X-Ray Inference System on Hadoop/Spark

A distributed inference system for multimodal DenseNet121 chest X-ray classification running on Apache Spark with HDFS integration.

## Features

- **Multimodal Deep Learning**: DenseNet121 with patient metadata fusion (age, gender, view position)
- **14 Disease Detection**: Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Pleural Thickening, Pneumonia, Fibrosis, Edema, Consolidation
- **GRAD-CAM Visualization**: Interpretable heatmaps showing model attention
- **Spark Distributed Processing**: Scalable batch inference
- **HDFS Storage**: Persistent storage for results
- **Flask Web UI**: Easy-to-use upload interface

## Project Structure

```
chest_xray_spark/
├── config.py           # Configuration
├── preprocessing.py    # Image/metadata preprocessing
├── model_loader.py     # TensorFlow model loading
├── inference.py        # Spark inference logic
├── gradcam.py          # GRAD-CAM heatmaps
├── hdfs_utils.py       # HDFS utilities
├── spark_job.py        # Main Spark entry point
└── flask_app/
    ├── app.py          # Flask application
    └── templates/      # HTML templates
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Flask App (Development)

```bash
python -m chest_xray_spark.flask_app.app
```

### 3. Submit Spark Job

```bash
./submit_spark_job.sh
```

## Configuration

Edit `chest_xray_spark/config.py` to configure:
- HDFS paths
- Spark settings
- Model location
- Image preprocessing parameters

## License

MIT
