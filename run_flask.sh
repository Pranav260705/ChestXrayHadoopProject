#!/bin/bash
# Run Flask Development Server

# Navigate to project directory
cd "$(dirname "$0")"

# Set environment variables
export USE_LOCAL_FS=false
export HDFS_CMD=/Users/aditchauhan/hadoop/bin/hdfs
export FLASK_ENV=development

echo "=========================================="
echo "Chest X-Ray Inference Web Application"
echo "=========================================="
echo ""
echo "USE_LOCAL_FS=$USE_LOCAL_FS"
if [ "$USE_LOCAL_FS" = "false" ]; then
    echo "Storage: HDFS (via CLI at $HDFS_CMD)"
else
    echo "Storage: Local filesystem"
fi
echo ""
echo "Starting Flask server on http://localhost:5001"
echo "Press Ctrl+C to stop"
echo ""

# Use the venv's Python directly with full path
"$(pwd)/venv/bin/python" -m chest_xray_spark.flask_app.app
