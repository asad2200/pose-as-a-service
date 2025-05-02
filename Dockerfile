FROM python:3.10-slim

# Never buffer Python output → instant logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # FastAPI will read this when run with Uvicorn / Gunicorn
    PORT=60000

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Python dependencies
WORKDIR /cloudpose

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Project source
COPY . .

# Expose port & run
EXPOSE ${PORT}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60000", "--workers", "1"]
