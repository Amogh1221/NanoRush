FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run FastAPI server on port 7860 with 5 workers for concurrency
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "5"]
