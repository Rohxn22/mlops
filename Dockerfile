# Stage 1: Builder
# Pulls data, trains model, runs tests
# This stage is discarded after build

FROM python:3.10-slim-buster AS builder

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

# permissions
RUN chmod +x /app/tests && \
    chmod +w /app/tests && \
    chmod +x /app/prediction_model && \
    mkdir -p /app/prediction_model/trained_models && \
    mkdir -p /app/prediction_model/datasets && \
    chmod +w /app/prediction_model/trained_models && \
    chmod +w /app/prediction_model/datasets

# Install dependencies (no DVC needed in runtime)
RUN pip install --no-cache-dir -r requirements.txt

# AWS credentials for DVC pull and MLflow
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

ENV PYTHONPATH="/app:/app/prediction_model"
ENV GIT_PYTHON_REFRESH=quiet

# Copy dataset (should be pulled by CI before Docker build)
COPY prediction_model/datasets/loan_data_part_1.csv /app/prediction_model/datasets/

# Train model (logs to MLflow)
RUN python /app/prediction_model/training_pipeline.py

# Run tests
RUN pytest -v /app/tests/test_prediction.py
RUN pytest --junitxml=/app/tests/test-results.xml /app/tests/test_prediction.py


# Stage 2: Runtime
# Only what's needed to serve the FastAPI app
# No DVC, no build tools = smaller image

FROM python:3.10-slim-buster AS runtime

RUN pip install --upgrade pip

WORKDIR /app

# Copy only application code from builder
COPY --from=builder /app/main.py .
COPY --from=builder /app/static ./static
COPY --from=builder /app/prediction_model ./prediction_model
COPY --from=builder /app/requirements.txt .

ENV PYTHONPATH="/app/prediction_model"
ENV GIT_PYTHON_REFRESH=quiet

# AWS credentials for MLflow access at runtime
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Install only runtime dependencies (no DVC)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir boto3

EXPOSE 8005

ENTRYPOINT ["python"]
CMD ["main.py"]