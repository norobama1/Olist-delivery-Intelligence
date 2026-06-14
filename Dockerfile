FROM python:3.11-slim

WORKDIR /app

# API-only deps: shap/matplotlib excluded to avoid numpy version conflict
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs at runtime
COPY app/    ./app/
COPY src/Main.py ./src/Main.py
COPY models/model_bundle.joblib ./models/model_bundle.joblib

ENV MODEL_PATH=models/model_bundle.joblib

EXPOSE 8000

CMD ["uvicorn", "src.Main:app", "--host", "0.0.0.0", "--port", "8000"]
