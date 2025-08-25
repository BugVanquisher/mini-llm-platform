# Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]