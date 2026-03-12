# 🏨 Wellness Tourism Prediction (End-to-End MLOps)

[![Tourism Project Pipeline](https://github.com/vk11797/Wellness-Tourism-Prediction-MLOps/actions/workflows/pipeline.yml/badge.svg)](https://github.com/vk11797/Wellness-Tourism-Prediction-MLOps/actions)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/VKblues2025/wellness-tourism)

An automated Machine Learning pipeline designed to predict whether a customer will purchase a wellness tourism package. This project demonstrates the full MLOps lifecycle: from automated data registration to CI/CD deployment on Hugging Face Spaces.

---

## 🏗️ Project Architecture

The project follows a modular MLOps workflow:
1. **Data Registration:** Raw CSV is uploaded to Hugging Face Datasets.
2. **Preprocessing:** Data is cleaned, split, and versioned (`train_v1.csv`, `test_v1.csv`).
3. **Model Training:** XGBoost model training with Hyperparameter Tuning (GridSearch) and Experiment Tracking via **MLflow**.
4. **Deployment:** Automated deployment of a **Streamlit** app to Hugging Face Spaces via Docker.



---

## 📂 Repository Structure

```text
├── .github/workflows/       # GitHub Actions CI/CD pipeline
├── data/                    # Local data storage (raw csv)
├── deployment/              # App files for HF Spaces (app.py, Dockerfile, README.md)
├── hosting/                 # Script to trigger HF Space deployment
├── model_building/          # Core ML scripts (prep.py, train.py, data_register.py)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

1. Prerequisites
Python 3.9+

Hugging Face Account & API Token

GitHub Secrets configured (HF_TOKEN)

2. Local Setup
Bash
# Clone the repo
git clone [https://github.com/vk11797/Wellness-Tourism-Prediction-MLOps.git](https://github.com/vk11797/Wellness-Tourism-Prediction-MLOps.git)

# Install dependencies
pip install -r requirements.txt
3. Running the Pipeline
The pipeline is fully automated via GitHub Actions. Simply push your changes to the main branch:

Bash
git add .
git commit -m "Improve model parameters"
git push origin main
📊 Experiment Tracking (MLflow)
We use MLflow to track model performance. During each training run, multiple trials are logged to monitor:

accuracy, f1-score, precision, recall

Hyperparameters: max_depth, learning_rate, n_estimators

🛠️ Technology Stack
Modeling: XGBoost, Scikit-Learn

Tracking: MLflow

Cloud/Hosting: Hugging Face (Model Hub, Dataset Hub, Spaces)

CI/CD: GitHub Actions

Containerization: Docker

Interface: Streamlit

👤 Author
VK - GitHub Profile
