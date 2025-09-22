# Short Tech Report

## 1. Results Obtained
- Built a **binary classification pipeline** to predict near-future machine failures using the Microsoft Azure Predictive Maintenance dataset.  
- Implemented **target engineering** with a gap (4h) and horizon (2h) to create preventive failure labels.  
- Developed a **fully containerized environment** using Docker Compose:
  - MLflow server for experiment tracking and artifact storage.
  - Prefect for orchestration of the end-to-end pipeline.
  - PostgreSQL databases for metadata persistence.
  - LocalStack for S3 emulation.
  - Inference container exposing a REST API.  
- Validated reproducibility: cloning the repo and running `docker compose up` executes the entire training pipeline and starts the inference service.  

---

## 2. Key Decisions
- **Problem framing:** shifted from multiclass (which component fails) to binary classification (will a failure occur) to align with the business focus on early warnings.  
- **Evaluation metric:** prioritized **F1 score** to balance precision and recall in an imbalanced dataset.  
- **Models selected:** Logistic Regression (baseline), LightGBM and CatBoost as main candidates.  
- **Reproducibility:** Docker + Prefect orchestration instead of manual Makefile targets, to guarantee one-command execution.  
- **design_notes:** design_notes.md file includes greater detail on all decisions rationale
---

## 3. Trade-offs
- **Simplicity vs. fidelity:** multiclass modeling was skipped, although some simultaneous failures exist (~0.5% of cases). Binary framing is easier to implement and evaluate.  
- **Time constraints:** linting, CI pre-checks, and continuous monitoring were left as roadmap items.  
- **Modeling complexity:** opted for tree-based gradient boosting models instead of neural networks to favor interpretability and fast iteration.  

---

## 4. Lessons Learned
- **Target engineering is crucial:** defining the gap and horizon was the most impactful step in aligning the model with the business use case.  
- **Reproducibility adds value:** having the full environment spin up with Docker Compose was highly effective for onboarding.  
- **Monitoring matters:** even if not fully implemented, Evidently demos highlighted the importance of drift analysis in predictive maintenance.  

---

## 5. Next Steps / Roadmap
- **Feature engineering:**  
  - Adjust pipeline to compute **rolling features first**, then merge with other datasets for efficiency.  
- **Handling imbalance:**  
  - Evaluate techniques like **SMOTE** or alternative resampling strategies.  
  - Consider implications carefully since this is time-series data and duplication may not be realistic.  
- **Model selection automation:**  
  - Introduce automatic selection based on validation scores or historical performance.  
- **Retraining strategy:**  
  - Move from one-off training to **scheduled periodic retraining** with Prefect.  
- **Monitoring:**  
  - Extend Evidently integration to continuous **data drift** and **model performance** monitoring.  
  - Add infrastructure health checks.  
- **Engineering maturity:**  
  - Integrate linting (black), pre-commit hooks, and CI/CD workflows (GitHub Actions).  
