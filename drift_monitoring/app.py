import streamlit as st
import streamlit.components.v1 as components
import boto3
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import datetime, timedelta
import sys
import os

# Add project root to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prediction_model.config import config

s3 = boto3.client('s3')

# S3 folder where batch prediction results are uploaded by the API
DRIFT_PREFIX = config.FOLDER + "/"

# Baseline: the dataset the model was trained on (stored in S3)
BASELINE_KEY = f"datasets/{config.DATASET_FILE}"


def list_recent_csv_files(max_days=7):
    """Look back up to max_days to find uploaded batch prediction CSVs."""
    for i in range(max_days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        folder = f"{DRIFT_PREFIX}{date}/"
        response = s3.list_objects_v2(Bucket=config.S3_BUCKET, Prefix=folder)
        files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
        if files:
            return files
    return []


def load_csv_from_s3(key):
    response = s3.get_object(Bucket=config.S3_BUCKET, Key=key)
    return pd.read_csv(response['Body'])


def main():
    st.title("Data Drift Monitor")
    st.caption(f"Baseline: {BASELINE_KEY} | Bucket: {config.S3_BUCKET}")

    # Load baseline (training data)
    try:
        baseline_df = load_csv_from_s3(BASELINE_KEY)
        baseline_df = baseline_df[config.FEATURES]  # keep only model features
    except Exception as e:
        st.error(f"Could not load baseline data from S3: {e}")
        return

    # Load recent batch prediction files
    csv_files = list_recent_csv_files()
    if not csv_files:
        st.warning("No batch prediction data found in S3 from the last 7 days.")
        st.info("Upload a CSV via the /batch_prediction API endpoint to generate drift data.")
        return

    selected_file = st.selectbox("Select a batch prediction file to compare:", csv_files)

    if selected_file:
        current_df = load_csv_from_s3(selected_file)
        current_df = current_df.drop(columns=['Prediction', config.TARGET], errors='ignore')
        current_df = current_df[config.FEATURES]  # align columns with baseline

        st.write(f"Baseline rows: {len(baseline_df)} | Current rows: {len(current_df)}")

        # Run Evidently drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline_df, current_data=current_df)

        report_path = "drift_report.html"
        report.save_html(report_path)

        with open(report_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=1000, scrolling=True)


if __name__ == "__main__":
    main()