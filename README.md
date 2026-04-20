# Machine Learning Operations (MLOps) - Loan Prediction V2

## MLOps Maturity Level 4

## Overview
This project implements a robust MLOps pipeline for loan prediction using a rich 15-feature dataset. The system facilitates continuous integration, deployment, and monitoring of machine learning models with AWS infrastructure, MLflow experiment tracking, and comprehensive monitoring.

## Key Features

### 🔄 **Data Versioning & Management**
- **DVC (Data Version Control)** for dataset versioning
- **S3 Integration** for scalable data storage
- **Rich Feature Dataset** with 15 loan-related features

### 🚀 **Continuous Integration (CI)**
- **GitHub Actions** workflow automation
- **Docker** containerization and image building
- **Pytest** automated testing
- **AWS ECR** for container registry

### 🧪 **Experiment Tracking & Model Management**
- **MLflow** for experiment tracking and model versioning
- **Hyperopt** for hyperparameter optimization
- **Model Registry** for production model management
- **XGBoost + Logistic Regression** model comparison

### 📦 **Continuous Deployment (CD)**
- **FastAPI** REST API for predictions
- **AWS EKS** Kubernetes deployment
- **Real-time** and **batch prediction** endpoints
- **Health monitoring** endpoints

### 📊 **Continuous Monitoring (CM)**
- **Prometheus** metrics collection via `/metrics` endpoint
- **Grafana** visualization dashboards
- **Resource monitoring** for Kubernetes clusters
- **API performance** tracking

### 🔄 **Continuous Training (CT)**
- **Automated retraining** on new data commits
- **GitHub Actions** triggered training pipelines
- **Model performance** validation

### 📈 **Drift Monitoring**
- **Streamlit** application for drift detection
- **Data drift** and **target drift** monitoring
- **Data quality** checks and validation

## API Endpoints

### Core Endpoints
- `GET /` - Frontend web interface
- `POST /prediction_api` - Single loan prediction
- `POST /batch_prediction` - Batch CSV predictions
- `GET /health` - System health and diagnostics
- `GET /metrics` - Prometheus monitoring metrics
- `GET /docs` - Interactive API documentation

## Architecture

The system follows MLOps best practices with:
- **Microservices** architecture
- **Container-first** deployment
- **Infrastructure as Code**
- **Automated CI/CD** pipelines
- **Comprehensive monitoring**

## Dataset Features

The enhanced loan prediction model uses 15 features:
- **Personal**: age, occupation_status, years_employed
- **Financial**: annual_income, credit_score, savings_assets, current_debt
- **Credit History**: credit_history_years, defaults_on_file, delinquencies_last_2yrs, derogatory_marks
- **Loan Details**: product_type, loan_intent, loan_amount, debt_to_income_ratio

## Technology Stack

- **ML**: scikit-learn, XGBoost, hyperopt
- **API**: FastAPI, Pydantic, uvicorn
- **Tracking**: MLflow, DVC
- **Infrastructure**: AWS (EKS, ECR, S3), Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana
- **CI/CD**: GitHub Actions
- **Frontend**: HTML/CSS/JavaScript

## Quick Start

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run locally**: `python main.py`
4. **Access API**: `http://localhost:8005`
5. **View docs**: `http://localhost:8005/docs`

## Production Deployment

The application is deployed on AWS EKS with:
- **Load balancing** for high availability
- **Auto-scaling** based on demand
- **Health checks** and monitoring
- **Secure secrets** management