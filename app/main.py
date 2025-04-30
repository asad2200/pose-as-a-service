import os
import logging
import traceback

import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from app.models import ImageRequest, AnnotatedImageResponse, PoseResponse
from app.utils import process_and_annotate_image, process_pose_detection

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pose Estimation API",
    description="RESTful API for detecting pose keypoints in images",
    version="1.0.0"
)

# Configure model path - can be set via environment variable
MODEL_PATH = os.getenv("MODEL_PATH", "models/movenet-full-256.tflite")

# Global variables
interpreter = None

@app.on_event("startup")
async def startup_event():
    global interpreter
    try:
        # Load the TensorFlow Lite model
        logger.info(f"Loading pose detection model from {MODEL_PATH}...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/pose/json", 
          summary="Detect pose keypoints in an image",
          description="Accepts a base64-encoded image and returns detected pose keypoints",
          response_description="Pose keypoints, bounding boxes, and timing metrics",
          response_model=PoseResponse)
async def detect_pose(request: ImageRequest):
    """Endpoint to detect pose in a base64 encoded image."""
    try:
        # Validate request ID
        if not request.id:
            raise HTTPException(status_code=400, detail="Missing ID in request")
            
        # Process the image
        try:
            result = await process_pose_detection(interpreter, request.image, request.id)
            return PoseResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/pose/image", 
          summary="Detect pose keypoints and return annotated image",
          description="Accepts a base64-encoded image and returns the annotated image with pose keypoints",
          response_description="Base64-encoded annotated image with pose keypoints",
          response_model=AnnotatedImageResponse)
async def detect_pose_with_image(request: ImageRequest):
    """Endpoint to detect pose in a base64 encoded image and return annotated image."""
    try:
        # Validate request ID
        if not request.id:
            raise HTTPException(status_code=400, detail="Missing ID in request")
            
        # Process and annotate the image
        try:
            base64_annotated_img = await process_and_annotate_image(interpreter, request.image)
            return AnnotatedImageResponse(id=request.id, image=base64_annotated_img)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health", 
         summary="Check API health",
         description="Verify if the API is running and the model is loaded",
         response_description="Health status of the API")
async def health_check():
    """Endpoint to check API health."""
    return {"status": "healthy", "model_loaded": interpreter is not None}

if __name__ == "__main__":
    import uvicorn
    
    # Choose a port between 60000-61000
    PORT = int(os.getenv("PORT", "60000"))
    
    # Run the FastAPI application
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)