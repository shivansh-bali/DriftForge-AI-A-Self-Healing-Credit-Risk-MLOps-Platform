# 🚀 DriftForge AI

### A Self-Healing Credit Risk MLOps Platform

An end-to-end **production-grade machine learning system** that detects data drift, monitors model performance, and automatically retrains and redeploys improved models.

Built to simulate how **real-world ML systems operate in fintech and enterprise environments**.

---

# 📌 Overview

DriftForge AI is not just a model — it is a **complete ML lifecycle platform**.

It includes:

* Real-time prediction API
* Production data logging
* Data & concept drift detection
* Automated retraining pipeline
* Model versioning & registry
* MLflow experiment tracking
* Monitoring dashboard
* Frontend control console
* CI/CD automation

---

# 🧭 System Architecture

```
Frontend UI → FastAPI → ML Model
                    ↓
                Logging
                    ↓
          Monitoring Pipelines
        (Drift + Performance)
                    ↓
            Trigger Engine
                    ↓
          Retraining Pipeline
                    ↓
            Model Versioning
                    ↓
                MLflow
                    ↓
          Monitoring Dashboard
```

---

# ⚙️ Key Features

## 🔮 Prediction API

* Built with FastAPI
* Real-time credit default prediction
* Returns probability + risk classification

---

## 📝 Production Logging

* Logs every prediction
* Stores inputs, outputs, timestamps
* Enables monitoring and retraining

---

## 📉 Data Drift Detection

* Detects feature distribution changes
* Computes drifted feature %
* Signals model degradation

---

## 📊 Performance Monitoring

* Tracks ROC-AUC and Recall
* Detects concept drift
* Uses production outcomes

---

## 🤖 Auto-Retraining System

* Combines historical + production data
* Retrains candidate model
* Compares against production model
* Promotes only if performance improves

---

## 🗂️ Model Versioning

* Versioned models (`model_v1`, `model_v2`, …)
* Registry-based production pointer

---

## 🧪 MLflow Experiment Tracking

* Logs parameters, metrics, artifacts
* Tracks retraining runs
* Enables auditability

---

## 🖥️ Monitoring Dashboard

* Drift status
* Performance metrics
* Retraining history
* Model version display

---

## 🌐 Frontend Control Console

* Input loan details
* View prediction + risk
* Trigger monitoring & retraining

---

## 🔁 CI/CD Pipeline

* Automated monitoring validation
* Retraining checks
* Frontend build
* Docker packaging

---

# 🧱 Tech Stack

| Layer            | Technology       |
| ---------------- | ---------------- |
| Backend          | FastAPI          |
| ML               | Scikit-learn     |
| Monitoring       | Evidently        |
| Tracking         | MLflow           |
| Dashboard        | Streamlit        |
| Frontend         | React + Tailwind |
| Automation       | Python           |
| CI/CD            | GitHub Actions   |
| Containerization | Docker           |

---

# 📂 Project Structure

```
ml-drift-system/
│
├── api/                  # FastAPI service
├── monitoring/           # Drift & performance pipelines
├── retraining/           # Trigger + retrain pipeline
├── dashboard/            # Streamlit dashboard
├── frontend/             # React UI
├── models/               # Model versions
├── logs/                 # Prediction logs
├── run_system.py         # Orchestrator
├── requirements.txt
└── .github/workflows/    # CI/CD pipeline
```

---

# ▶️ How to Run

## 1️⃣ Start API

```bash
uvicorn api.main:app --reload
```

👉 http://127.0.0.1:8000/docs

---

## 2️⃣ Start MLflow

```bash
mlflow ui
```

👉 http://127.0.0.1:5000

---

## 3️⃣ Start Dashboard

```bash
streamlit run dashboard/app.py
```

👉 http://localhost:8501

---

## 4️⃣ Start Frontend

```bash
cd frontend
npm install
npm run dev
```

👉 http://localhost:5173

---

## 5️⃣ Run Monitoring + Retraining

```bash
python run_system.py
```

---

# 🔁 CI/CD Pipeline

Automated using GitHub Actions.

Pipeline includes:

* Dependency installation
* Monitoring validation
* Retraining trigger checks
* Frontend build
* Docker image creation

---

# 📊 Metrics Tracked

* Data drift status
* Drifted feature %
* ROC-AUC
* Recall
* Prediction distribution
* Retraining events

---

# 🧠 Real-World Relevance

This system mirrors production ML platforms used in:

* Banking risk modeling
* Loan underwriting
* Fraud detection
* Fintech ML systems

It demonstrates:

* Handling model decay
* Continuous monitoring
* Automated retraining
* Model governance

---

# 🚀 Future Enhancements

* Cloud deployment (AWS/GCP)
* Kubernetes orchestration
* Feature store integration
* Real-time monitoring
* Alerting system (Slack/Email)
* A/B model testing

---

# 👨‍💻 Author

Built as a full-stack MLOps system demonstrating:

* Production ML lifecycle design
* Monitoring & observability
* Automated retraining pipelines
* Experiment tracking
* Full-stack ML deployment

---

# ⭐ If You Like This Project

Star ⭐ the repo and feel free to connect or contribute!
