"""Main FastAPI application."""
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import Tuple, Optional
import uuid

from app.config import API_KEY
from app.schemas import (
    InferenceRequestSchema,
    InferenceResponseSchema,
    PersonResultSchema,
    TopKItemSchema,
    WindowSchema
)
from app.logging import log_inference_request, get_logger
from app.health import router as health_router

app = FastAPI(title="Cloud HAR API", version="1.0.0")

# Include health router
app.include_router(health_router)

logger = get_logger("main")


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """Dependency to verify API key."""
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


def mock_inference_logic(pose_conf: float) -> Tuple[str, float, list[TopKItemSchema]]:
    """
    Mock inference logic.
    
    Returns:
        tuple: (activity, confidence, top_k)
    """
    if pose_conf < 0.4:
        activity = "unknown"
        confidence = 0.2
        top_k = [
            TopKItemSchema(label="unknown", score=0.2),
            TopKItemSchema(label="standing", score=0.1),
            TopKItemSchema(label="walking", score=0.1),
        ]
    else:
        activity = "standing"
        confidence = 0.6
        top_k = [
            TopKItemSchema(label="standing", score=0.6),
            TopKItemSchema(label="walking", score=0.2),
            TopKItemSchema(label="sitting", score=0.2),
        ]
    
    return activity, confidence, top_k


@app.post("/v1/activity/infer", response_model=InferenceResponseSchema)
async def infer_activity(
    request: InferenceRequestSchema,
    api_key: str = Depends(verify_api_key)
) -> InferenceResponseSchema:
    """
    Inference endpoint for activity recognition.
    
    Accepts skeleton window data and returns mock activity predictions.
    """
    request_id = str(uuid.uuid4())
    
    with log_inference_request(
        device_id=request.device_id,
        camera_id=request.camera_id,
        num_people=len(request.people),
        request_id=request_id
    ):
        # Process each person
        results = []
        for person in request.people:
            activity, confidence, top_k = mock_inference_logic(person.pose_conf)
            
            result = PersonResultSchema(
                track_id=person.track_id,
                activity=activity,
                confidence=confidence,
                top_k=top_k
            )
            results.append(result)
        
        # Build response
        response = InferenceResponseSchema(
            schema_version=1,
            device_id=request.device_id,
            camera_id=request.camera_id,
            window=request.window,
            results=results
        )
        
        return response
