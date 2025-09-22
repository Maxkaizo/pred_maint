#!/bin/bash
echo "Waiting for model catboost_pred_maintenance with tag stage=Staging..."
python wait_for_model.py

echo "Starting inference API..."
exec gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --timeout 120 \
    --workers 2
