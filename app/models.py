from typing import List
from pydantic import BaseModel

class ImageRequest(BaseModel):
    """Request model for pose detection API."""
    id: str
    image: str  # Base64 encoded image

class AnnotatedImageResponse(BaseModel):
    """Response model for annotated image API."""
    id: str
    image: str  # Base64 encoded annotated image

class BoundingBox(BaseModel):
    """Model for bounding box information."""
    x: float
    y: float
    width: float
    height: float
    probability: float

class PoseResponse(BaseModel):
    """Response model for pose detection API."""
    id: str
    count: int
    boxes: List[BoundingBox]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float