# ─────────────────────────────────────────────
# STAGE 1: Builder
# Purpose: train the model and run tests
# This stage is thrown away after build — only
# the trained artifacts move to Stage 2
# ─────────────────────────────────────────────
FROM python:3.10-slim-buster AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset
COPY prediction_model/ ./prediction_model/
COPY tests/ ./tests/
COPY main.py .
COPY static/ ./static/

# Dataset is pulled from S3 via DVC in CI before this build runs
COPY prediction_model/datasets/loan_data_part_2.csv ./prediction_model/datasets/

# AWS credentials passed in from CI secrets (needed for MLflow logging)
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV PYTHONPATH="/app"

# Train the model — logs metrics and registers model in MLflow
RUN python prediction_model/training_pipeline.py

# Run tests — if this fails, the image will NOT be built or pushed
RUN pytest -v tests/test_prediction.py


# ─────────────────────────────────────────────
# STAGE 2: Runtime
# Purpose: serve the FastAPI prediction API
# Only contains what's needed to run the app —
# no training code, no datasets, no test files
# ─────────────────────────────────────────────
FROM python:3.10-slim-buster AS runtime

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy only the files needed to serve predictions
COPY --from=builder /app/main.py .
COPY --from=builder /app/static ./static
COPY --from=builder /app/prediction_model/__init__.py ./prediction_model/
COPY --from=builder /app/prediction_model/predict.py ./prediction_model/
COPY --from=builder /app/prediction_model/config ./prediction_model/config
COPY --from=builder /app/prediction_model/processing ./prediction_model/processing

# Copy the trained model — baked into the image, no MLflow needed at runtime
COPY --from=builder /app/prediction_model/trained_models/model.pkl ./prediction_model/trained_models/

# AWS credentials for MLflow model loading at runtime
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV PYTHONPATH="/app"

EXPOSE 8005

CMD ["python", "main.py"]