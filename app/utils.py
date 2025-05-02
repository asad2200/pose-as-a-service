import base64
import logging
import traceback
import time
import cv2
import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from io import BytesIO

logger = logging.getLogger(__name__)

def decode_base64_image(base64_string: str):
    """Decode base64 string to image."""
    try:
        # Remove header if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {str(e)}")

def process_image(interpreter, image, request_id: str):
    """Process image to detect poses."""
    try:
        start_preprocess = time.time()
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get image dimensions
        img_height, img_width, _ = image.shape
        
        # Resize image to match model input
        input_shape = input_details[0]['shape'][1:3]  # Expected input size
        resized_image = cv2.resize(image, input_shape)
        
        # Normalize the image
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0
        end_preprocess = time.time()
        
        # Perform inference
        start_inference = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Extract keypoints
        keypoints_output = interpreter.get_tensor(output_details[0]['index'])  # Shape: (1, 1, 17, 3)
        keypoints = keypoints_output[0][0] # Shape: (17, 3) - 17 keypoints, (y, x, confidence)
        end_inference = time.time()
        
        # Post-process results
        start_postprocess = time.time()
        
        # For this implementation, we'll detect one person
        # In a production environment, you would enhance this to detect multiple persons
        count = 1
        
        # Process keypoints
        keypoints_list = []
        person_keypoints = []
        
        for kp in keypoints:
            y, x, confidence = kp
            person_keypoints.append([float(x), float(y), float(confidence)])
        
        keypoints_list.append(person_keypoints)
        
        # Create a bounding box from keypoints
        x_coords = [kp[0] for kp in person_keypoints if kp[2] > 0.1]
        y_coords = [kp[1] for kp in person_keypoints if kp[2] > 0.1]
        
        if x_coords and y_coords:
            min_x = min(x_coords)
            min_y = min(y_coords)
            width = max(x_coords) - min_x
            height = max(y_coords) - min_y
            probability = float(np.mean([kp[2] for kp in person_keypoints]))
        else:
            # Fallback if no valid keypoints
            min_x, min_y = 0, 0
            width, height = img_width, img_height
            probability = 0.1
        
        boxes = [
            {
                "x": float(min_x),
                "y": float(min_y),
                "width": float(width),
                "height": float(height),
                "probability": probability
            }
        ]
        
        end_postprocess = time.time()
        
        # Calculate timings
        speed_preprocess = end_preprocess - start_preprocess
        speed_inference = end_inference - start_inference
        speed_postprocess = end_postprocess - start_postprocess
        
        return {
            "id": request_id,
            "count": count,
            "boxes": boxes,
            "keypoints": keypoints_list,
            "speed_preprocess": speed_preprocess,
            "speed_inference": speed_inference,
            "speed_postprocess": speed_postprocess
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Image processing error: {str(e)}")

def annotate_image(image, keypoints_output):
    """Create annotated image with detected pose keypoints."""
    try:
        # Create a copy of the original image
        annotated_image = image.copy()
        
        # Get image dimensions
        img_height, img_width, _ = image.shape
        
        # Define connections between keypoints for visualization
        connections = [
            (5, 6), (5, 11), (6, 12), (11, 12),      # Shoulders and hips
            (5, 7), (7, 9), (6, 8), (8, 10),         # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (0, 1), (0, 2), (1, 3), (2, 4),          # Face
            (0, 5), (0, 6)                           # Neck
        ]
        
        # Draw each keypoint
        for person_keypoints in keypoints_output:
            for kp in person_keypoints:
                x, y, confidence = kp
                # if confidence > 0.2:  # Only draw points with reasonable confidence
                x_coord = int(x * img_width)
                y_coord = int(y * img_height)
                # Draw circle at keypoint location (green)
                cv2.circle(annotated_image, (x_coord, y_coord), 5, (0, 255, 0), -1)
            
            # Draw connections between keypoints
            for connection in connections:
                start_idx, end_idx = connection
                x1, y1, c1 = person_keypoints[start_idx]
                x2, y2, c2 = person_keypoints[end_idx]
                
                # if c1 > 0.2 and c2 > 0.2:  # Only draw connections with reasonable confidence
                x1_coord = int(x1 * img_width)
                y1_coord = int(y1 * img_height)
                x2_coord = int(x2 * img_width)
                y2_coord = int(y2 * img_height)
                
                # Draw line connecting keypoints (red)
                cv2.line(annotated_image, (x1_coord, y1_coord), (x2_coord, y2_coord), (0, 0, 255), 2)
        
        return annotated_image
    except Exception as e:
        logger.error(f"Error annotating image: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Image annotation error: {str(e)}")

def encode_image_to_base64(image):
    """Encode an image to base64 string."""
    try:
        # Encode the image to JPG format
        _, buffer = cv2.imencode('.jpg', image)
        # Convert to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise ValueError(f"Image encoding error: {str(e)}")

async def process_and_annotate_image(interpreter, base64_image: str):
    """Process a base64 encoded image and return annotated image in base64 format."""
    try:
        # Decode the base64 image
        image = decode_base64_image(base64_image)
        
        # Process image to get keypoints
        result = process_image(interpreter, image, "temp_id")  # ID not needed for annotation
        
        # Annotate the image with keypoints
        annotated_img = annotate_image(image, result["keypoints"])
        
        # Encode the annotated image to base64
        base64_annotated_img = encode_image_to_base64(annotated_img)
        
        return base64_annotated_img
    except Exception as e:
        logger.error(f"Error in process_and_annotate_image: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Image processing and annotation error: {str(e)}")

async def process_pose_detection(interpreter, base64_image: str, request_id: str):
    """Process a base64 encoded image and return pose detection results."""
    try:
        # Decode the base64 image
        image = decode_base64_image(base64_image)
        
        # Process image to get keypoints and other results
        result = process_image(interpreter, image, request_id)
        
        return result
    except Exception as e:
        logger.error(f"Error in process_pose_detection: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Pose detection error: {str(e)}")