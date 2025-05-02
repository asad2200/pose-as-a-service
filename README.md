# CloudPose

CloudPose is a high-performance, containerized pose estimation service built with FastAPI, TensorFlow Lite, and OpenCV. It exposes RESTful endpoints for JSON‐based and image‐based inference, supports load testing via Locust, and can be deployed locally, in Docker, or on Kubernetes.

## Prerequisites

* **Python** 3.10 or newer
* **pip** package manager
* **Docker** (optional, for container builds)
* **kubectl** & a **Kubernetes** cluster (optional)
* **Locust** (for load testing)

---

## Installation

**Install dependencies**

   ```bash
   python -m venv venv            # optional but recommended
   source venv/bin/activate       # Linux/macOS
   # .\venv\Scripts\activate    # Windows PowerShell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running Locally

1. **Start the API server**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 60000
   ```

2. **Verify endpoints**

    ```bash
    curl -X 'POST' \
    'http://207.211.146.117:30000/pose/json' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": <base64-image-string>, "images": <base64-image-string>}'
    ```

    ```bash
    curl -X 'POST' \
    'http://207.211.146.117:30000/pose/image' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": <base64-image-string>, "image": <base64-image-string>}'
    ```

---

## Running Load Tests
1. **Run Locust (headless)**

   ```bash
   locust -f experiments/locustfile.py \
     --host http://localhost:60000 \
     --users 100 --spawn-rate 10 --run-time 1m
   ```
   
---

## Docker Usage

1. **Build the Docker image**

   ```bash
   docker build -t cloudpose:latest .
   ```

2. **Run the container**

   ```bash
   docker run -d -p 60000:60000 --name cloudpose cloudpose:latest
   ```

3. **Test the service**

   ```bash
   curl http://localhost:60000/pose/json
   ```

---

## Kubernetes Deployment

1. **Apply manifests**

   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

2. **Check rollout status**

   ```bash
   kubectl -n cloudpose get pods,svc
   ```

3. **Access via NodePort**

   * The service exposes port `60000` as a NodePort; query any node IP:

     ```bash
     curl http://<NODE_IP>:<NODE_PORT>/pose/json
     ```
