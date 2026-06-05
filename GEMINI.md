# GEMINI.md - Predictive Maintenance Project

## Project Overview
This project is an end-to-end Machine Learning Engineering solution for Predictive Maintenance, based on the Microsoft Azure Predictive Maintenance dataset. It aims to predict machine failures using binary classification models (CatBoost and LightGBM).

The project emphasizes MLOps best practices, including pipeline orchestration, experiment tracking, model versioning, and containerized deployment.

### Tech Stack
- **Languages:** Python 3.14 (via UV)
- **Orchestration:** Prefect 3.7
- **Experiment Tracking:** MLflow 3.13
- **Infrastructure:** Docker Compose, LocalStack (S3 emulation), PostgreSQL 14
- **ML Frameworks:** CatBoost 1.2.10, LightGBM 4.6.0, Scikit-learn, Pandas
- **Inference:** FastAPI, Uvicorn

## Core Architecture
1.  **Orchestration (Prefect):** Manages the full pipeline from bucket creation to model promotion.
2.  **Experiment Tracking & Model Registry (MLflow):** Logs parameters, metrics (AP, F1), and artifacts. Models are registered and tagged (e.g., `stage=Staging`).
3.  **Storage (LocalStack):** Emulates AWS S3 for datalake and MLflow artifacts.
4.  **Inference (FastAPI):** Dynamically loads the latest "Staging" model from MLflow for real-time predictions.

## Building and Running

### Prerequisites
- Docker and Docker Compose
- UV (for local development)
- A `.env` file at the root (see `README.MD` for template)

### Key Commands
- **Start Environment:**
  ```bash
  docker compose up --build
  ```
- **Trigger Pipeline (inside runner container):**
  The pipeline registers and runs automatically via `entrypoint.sh` on startup. To run manually:
  ```bash
  uv run python app/flows/main_pipeline.py
  uv run prefect deployment run "main_pipeline/pred-maintenance-pipeline"
  ```
- **Test Inference:**
  ```bash
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d @sample.json
  ```

## Development Conventions

### Project Structure
- `app/flows/`: Prefect orchestration logic.
- `app/tasks/`: Modular, reusable pipeline components.
- `inference_app/`: FastAPI service for serving predictions.
- `docs/`: Technical documentation (TDD, Design Notes).
- `notebooks/`: EDA and experimentation.

### Environment Management
The project uses **UV**. Configuration is managed in `pyproject.toml` and dependencies are pinned in `uv.lock`.

### MLflow Model Promotion
The final training task (`app/tasks/train_catboost_final.py`) automatically registers the model and sets the `stage` tag to `Staging`. The inference app loads the model based on this tag.

### Coding Style
- Follow PEP 8 standards.
- Use Prefect `@task` and `@flow` decorators for pipeline components.
- Ensure all artifacts are logged to MLflow within the pipeline.

## Development Status & Roadmap
- [x] Full training/inference integration.
- [x] Experiment tracking with MLflow.
- [x] Modular task-based architecture.
- [ ] CI/CD with GitHub Actions (TODO).
- [ ] Retraining schedules in Prefect (TODO).
- [ ] Monitoring with Evidently (TODO).
