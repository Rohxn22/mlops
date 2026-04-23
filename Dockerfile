# Stage 1: Builder
# Pulls data, trains model, runs tests
# This stage is discarded after build - size doesn't matter

FROM python:3.10-slim-buster AS builder

RUN pip install --upgrade pip

WORKDIR /app

# Copy only necessary files for training (not everything)
COPY requirements.txt .
COPY prediction_model/ ./prediction_model/
COPY tests/ ./tests/
COPY main.py .
COPY static/ ./static/

# Install all dependencies (including training deps)
RUN pip install --no-cache-dir -r requirements.txt

# AWS credentials for MLflow logging
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

ENV PYTHONPATH="/app:/app/prediction_model"
ENV GIT_PYTHON_REFRESH=quiet

# Copy dataset (should be pulled by CI before Docker build)
COPY prediction_model/datasets/loan_data_part_2.csv /app/prediction_model/datasets/

# Train model (logs to MLflow)
RUN python /app/prediction_model/training_pipeline.py

# Run tests
RUN pytest -v /app/tests/test_prediction.py


# Stage 2: Runtime
# Only what's needed to serve the FastAPI app
# Excludes: training libs (hyperopt, xgboost), datasets, tests, training scripts

FROM python:3.10-slim-buster AS runtime

RUN pip install --upgrade pip

WORKDIR /app

# Copy only essential application files
COPY --from=builder /app/main.py .
COPY --from=builder /app/static ./static

# Copy only runtime prediction code (not training code)
COPY --from=builder /app/prediction_model/__init__.py ./prediction_model/
COPY --from=builder /app/prediction_model/predict.py ./prediction_model/
COPY --from=builder /app/prediction_model/config ./prediction_model/config
COPY --from=builder /app/prediction_model/trained_models ./prediction_model/trained_models

# Copy optimized runtime requirements
COPY requirements-runtime.txt .

ENV PYTHONPATH="/app"
ENV GIT_PYTHON_REFRESH=quiet

# AWS credentials for MLflow access at runtime
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Install only runtime dependencies (excludes hyperopt, xgboost training, pytest)
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Clean up to reduce image size further
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/* && \
    find /usr/local/lib/python3.10 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10 -name "__pycache__" -type d -exec rm -rf {} + || true

EXPOSE 8005

ENTRYPOINT ["python"]
CMD ["main.py"]