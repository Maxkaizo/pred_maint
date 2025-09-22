#!/bin/bash
set -e

echo "Registering deployment..."
conda run -n walmart python flows/main_pipeline.py &

echo "Waiting 5 seconds for Prefect to process the deployment..."
sleep 5

echo "Running deployment..."
conda run -n walmart prefect deployment run "main_pipeline/pred-maintenance-pipeline"

echo "Deployment registered and executed."
# Keep container alive
tail -f /dev/null
