#!/bin/bash
# Submit Spark Job for Batch Inference

# Configuration
SPARK_MASTER="${SPARK_MASTER:-local[*]}"
INPUT_PATH="${1:-/chest_xray/uploads}"
OUTPUT_PATH="${2:-/chest_xray/results}"

# Navigate to project directory
cd "$(dirname "$0")"

echo "=========================================="
echo "Chest X-Ray Inference Spark Job"
echo "=========================================="
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Master: $SPARK_MASTER"
echo ""

# Create ZIP file of the package for distribution
if [ ! -f "chest_xray_spark.zip" ] || [ "$0" -nt "chest_xray_spark.zip" ]; then
    echo "Creating package archive..."
    zip -r chest_xray_spark.zip chest_xray_spark -x "*.pyc" -x "*__pycache__*" -x "*.egg-info*"
fi

# Submit Spark job
spark-submit \
    --master "$SPARK_MASTER" \
    --deploy-mode client \
    --conf spark.executor.memory=8g \
    --conf spark.executor.cores=4 \
    --conf spark.driver.memory=4g \
    --conf spark.sql.execution.arrow.pyspark.enabled=true \
    --py-files chest_xray_spark.zip \
    chest_xray_spark/spark_job.py \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    "${@:3}"

echo ""
echo "Job completed!"
