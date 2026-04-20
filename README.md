# Machine Learning Operations (MLOps) - Loan Prediction V2

## MLOps Maturity Level 4

## Overview
This project implements a robust MLOps pipeline for loan prediction using a rich 15-feature dataset. The system facilitates continuous integration, deployment, and monitoring of machine learning models with AWS infrastructure, MLflow experiment tracking, and comprehensive monitoring.

## Key Features

### 🔄 **Data Versioning & Management**
- **DVC (Data Version Control)** for dataset versioning
- **AWS S3** for scalable data storage
- **Rich Feature Dataset** with 15 loan-related features

### 🚀 **Continuous Integration (CI)**
- **GitHub Actions** workflow automation
- **Docker** containerization and image building
- **Docker Hub** for container registry
- **Pytest** automated testing

### 🧪 **Experiment Tracking & Model Management**
- **MLflow** for experiment tracking and model versioning
- **Hyperopt** for hyperparameter optimization
- **Model Registry** for production model management
- **XGBoost + Logistic Regression** model comparison

### 📦 **Continuous Deployment (CD)**
- **FastAPI** REST API for predictions
- **AWS EC2** for application hosting
- **Docker** containerized deployment
- **Real-time** and **batch prediction** endpoints
- **Health monitoring** endpoints

### 📊 **Continuous Monitoring (CM)**
- **Prometheus** metrics collection via `/metrics` endpoint
- **Grafana** visualization dashboards
- **cAdvisor** for container monitoring
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
- **Containerized** deployment on AWS EC2
- **Docker Hub** for image registry
- **GitHub Actions** for CI/CD automation
- **DVC + S3** for data versioning
- **MLflow** for experiment tracking
- **Comprehensive monitoring** stack

## Dataset Features

The enhanced loan prediction model uses 15 features:
- **Personal**: age, occupation_status, years_employed
- **Financial**: annual_income, credit_score, savings_assets, current_debt
- **Credit History**: credit_history_years, defaults_on_file, delinquencies_last_2yrs, derogatory_marks
- **Loan Details**: product_type, loan_intent, loan_amount, debt_to_income_ratio

## Technology Stack

### **Machine Learning**
- scikit-learn, XGBoost, hyperopt
- MLflow for experiment tracking
- DVC for data versioning

### **API & Web**
- FastAPI, Pydantic, uvicorn
- HTML/CSS/JavaScript frontend

### **Infrastructure**
- **AWS EC2** - Application hosting
- **AWS S3** - Data storage
- **Docker Hub** - Container registry
- **Docker** - Containerization

### **CI/CD & Monitoring**
- **GitHub Actions** - Automation
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **cAdvisor** - Container monitoring

## Deployment Pipeline

1. **Code Push** → GitHub repository
2. **GitHub Actions** triggers:
   - DVC data pull from S3
   - Docker image build
   - Push to Docker Hub
   - Deploy to EC2 via SSH
3. **EC2 Deployment**:
   - Pull latest image
   - Stop/remove old container
   - Start new container with environment variables

## Quick Start

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run locally**: `python main.py`
4. **Access API**: `http://localhost:8005`
5. **View docs**: `http://localhost:8005/docs`

## Production URLs

- **Application**: `http://ec2-13-62-110-20.eu-north-1.compute.amazonaws.com:8005`
- **MLflow**: `http://ec2-13-63-102-32.eu-north-1.compute.amazonaws.com:5000`
- **Monitoring**: Prometheus + Grafana stack