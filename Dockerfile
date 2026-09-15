FROM python:3.12-slim

WORKDIR /app

# Build deps for scipy/numpy wheels that need compiling on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data data/captures

EXPOSE 8080

# Bind to all interfaces inside the container; the dashboard's own auth token
# still guards every endpoint that can move the robot. Autoreload is a
# development-only feature and is not enabled here.
CMD ["python", "-m", "uvicorn", "src.dashboard.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
