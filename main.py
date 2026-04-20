from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import io
import boto3
import os
from datetime import datetime
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from prediction_model.predict import generate_predictions, generate_predictions_batch
from prediction_model.config import config


def upload_to_s3(file_content, filename):
    s3 = boto3.client('s3')
    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = f"{config.FOLDER}/{current_date}"
    filename_with_datetime = f"{filename}_{current_datetime}.csv"
    s3_key = f"{folder_path}/{filename_with_datetime}"
    s3.put_object(Bucket=config.S3_BUCKET, Key=s3_key, Body=file_content)
    return s3_key


mlflow.set_tracking_uri(config.TRACKING_URI)


app = FastAPI(
    title="Loan Prediction App V2 - MLOps",
    description="Enhanced loan prediction with rich feature set",
    version='2.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

Instrumentator().instrument(app).expose(app)
app.mount("/static", StaticFiles(directory="static"), name="static")


class LoanPrediction(BaseModel):
    age: int
    occupation_status: str
    years_employed: float
    annual_income: int
    credit_score: int
    credit_history_years: float
    savings_assets: int
    current_debt: int
    defaults_on_file: int
    delinquencies_last_2yrs: int
    derogatory_marks: int
    product_type: str
    loan_intent: str
    loan_amount: int
    debt_to_income_ratio: float


@app.get("/health")
def health_check():
    try:
        # Test MLflow connection
        mlflow.set_tracking_uri(config.TRACKING_URI)
        
        # Check if target experiment exists
        experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
        if experiment:
            mlflow_status = f"OK - Experiment '{config.EXPERIMENT_NAME}' found (ID: {experiment.experiment_id})"
            
            # Check for runs in the experiment
            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
            runs_count = len(runs_df)
            mlflow_status += f" - {runs_count} runs found"
        else:
            mlflow_status = f"Warning - Experiment '{config.EXPERIMENT_NAME}' not found, checking default experiment"
            
            # Check default experiment (Training runs)
            runs_df = mlflow.search_runs(experiment_ids=["0"], max_results=5)
            runs_count = len(runs_df)
            mlflow_status += f" - {runs_count} runs in default experiment"
            
    except Exception as e:
        mlflow_status = f"Error: {str(e)}"
    
    # Test local model file
    local_model_path = os.path.join(config.PACKAGE_ROOT, 'trained_models', 'loan_model_v2.pkl')
    local_model_status = "OK" if os.path.exists(local_model_path) else "Not found"
    
    # Test prediction function
    try:
        from prediction_model.predict import _load_best_model
        
        # Capture detailed debug output
        import sys
        from io import StringIO
        
        # Redirect stdout to capture debug prints
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            model = _load_best_model()
            debug_output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Try to get model info from MLflow
        experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
        if experiment:
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['metrics.f1_score DESC'],
                max_results=5
            )
            
            # Check for tagged best model
            best_tagged = runs_df[runs_df['tags.model_status'] == 'BEST_MODEL']
            if not best_tagged.empty:
                current_model = best_tagged.iloc[0]
                model_info = f"🏆 BEST_MODEL: {current_model.get('params.model_type', 'Unknown')} (F1: {current_model.get('metrics.f1_score', 0):.4f})"
            else:
                current_model = runs_df.iloc[0] if not runs_df.empty else None
                if current_model is not None:
                    model_info = f"Highest F1: {current_model.get('params.model_type', 'Unknown')} (F1: {current_model.get('metrics.f1_score', 0):.4f})"
                else:
                    model_info = "No model info available"
        else:
            model_info = "Experiment not found"
            
        prediction_status = f"OK - Model loaded successfully - {model_info}"
        if debug_output.strip():
            prediction_status += f" | Debug: {debug_output.strip()}"
            
    except Exception as e:
        # Capture any debug output even on failure
        debug_output = captured_output.getvalue() if 'captured_output' in locals() else ""
        error_msg = str(e)
        if debug_output.strip():
            error_msg += f" | Debug: {debug_output.strip()}"
        prediction_status = f"Error: {error_msg}"
    
    return {
        "status": "healthy",
        "mlflow": mlflow_status,
        "local_model": local_model_status,
        "prediction_service": prediction_status,
        "config": {
            "tracking_uri": config.TRACKING_URI,
            "experiment_name": config.EXPERIMENT_NAME,
            "model_name": config.MODEL_NAME
        }
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/prediction_api")
def predict_api(loan_details: LoanPrediction):
    data = loan_details.model_dump()
    prediction = generate_predictions([data])["prediction"][0]
    pred = "Approved" if prediction == "Y" else "Rejected"
    return {"status": pred}


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), index_col=False)
    
    # Check if all required features are present
    missing_cols = set(config.FEATURES) - set(df.columns)
    if missing_cols:
        return {"error": f"CSV file missing required columns: {list(missing_cols)}"}
    
    predictions = generate_predictions_batch(df)["prediction"]
    df['Prediction'] = predictions
    result = df.to_csv(index=False)
    
    # Upload to S3
    upload_to_s3(result.encode('utf-8'), file.filename)
    
    return StreamingResponse(
        io.BytesIO(result.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)