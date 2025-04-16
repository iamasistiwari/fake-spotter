from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from pathlib import Path
import requests
import uuid
import os
import asyncio
import concurrent.futures
from functools import lru_cache
import time
import cv2 
from model import FakeSpotter
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz
import hashlib
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


MODEL_SECRET = os.getenv("MODEL_TOKEN_SECRET")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins
    allow_credentials=False,        # Must be False when allow_origins=["*"]
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all headers
)


BASE_PATH = Path(__file__).resolve().parent
MODEL_DIR = BASE_PATH / "model_checkpoints"
VIDEO_DIR = BASE_PATH / "videos"
VIDEO_DIR.mkdir(exist_ok=True, mode=0o755)  # Ensure proper permissions

# Global model cache
model_cache = {}

class VideoRequest(BaseModel):
    video_url: HttpUrl
    token: str
    method: str


def is_token_valid(token: str) -> bool:
    try:
        india_timezone = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(india_timezone)
        timestamp = current_time_ist.day + current_time_ist.hour

        expected_string = str(timestamp) + MODEL_SECRET
        expected_hash = hashlib.sha256(expected_string.encode()).hexdigest()
        print("EXPECTED HASH", expected_hash)
        return expected_hash == token
    except Exception:
        return False

def download_video(url: str, output_path: Path):
    """Download video with simplified approach"""
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Write all content at once
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Verify the video
        verify_video(str(output_path))
        return output_path
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Video download failed: {e}")


def verify_video(video_path):
    """Verify that OpenCV can read the video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV cannot read the video file: {video_path}")
    
    # Read the first frame to ensure it's a valid video
    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read frames from video: {video_path}")
    
    cap.release()
    print(f"Video verified successfully: {video_path}")

def load_model(detector, model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    detector.load_model(str(model_path))

def get_loaded_model(model_path):
    """Get model from cache or load if not present"""
    model_path_str = str(model_path)
    if model_path_str not in model_cache:
        detector = FakeSpotter()
        load_model(detector, model_path)
        model_cache[model_path_str] = detector
    return model_cache[model_path_str]

def process_single_model(idx, model_path, video_path):
    """Process video with a single model"""
    try:
        # Ensure video is accessible before processing
        if not os.path.exists(video_path):
            return idx, {"error": f"Video file not found: {video_path}"}
            
        # Verify the video can be read
        try:
            verify_video(video_path)
        except RuntimeError as e:
            return idx, {"error": str(e)}
            
        # Use cached model
        detector = get_loaded_model(model_path)
        prediction = detector.predict(video_path)
        return idx, {
            "is_fake": prediction.get("is_fake"),
            "confidence": round(prediction.get("confidence", 0.0), 4)
        }
    except Exception as e:
        return idx, {"error": f"Processing error: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    model_files = [
        "fakespotter_epoch_20_train_acc_95_val_acc_85_auc_92.pth",
        "fakespotter_epoch_16_train_acc_95_val_acc_85_auc_92.pth",
        "fakespotter_epoch_6_train_acc_90_val_acc_85_auc_87.pth"
    ]
    
    print("Preloading models...")
    for model_file in model_files:
        model_path = MODEL_DIR / model_file
        if model_path.exists():
            detector = FakeSpotter()
            load_model(detector, model_path)
            model_cache[str(model_path)] = detector
            print(f"Preloaded model: {model_file}")
    print("Model preloading complete")

@app.post("/predict")
async def predict(body: VideoRequest, background_tasks: BackgroundTasks):
    print("Received:", body.dict())

    if not is_token_valid(body.token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    video_url = body.video_url
    method = body.method

    print("REQUEST RECEIVED")
    video_filename = f"{uuid.uuid4()}.mp4"
    video_path = VIDEO_DIR / video_filename

    try:
        # Step 1: Download and verify video
        await asyncio.to_thread(download_video, video_url, video_path)

        # Add a small delay to ensure file system has completed writing
        await asyncio.sleep(0.5)

        # Ensure the file exists and has size before proceeding
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Video download failed or file is empty: {video_path}")

        print(f"Video downloaded successfully: {video_path}")

        if method == "deep":
            model_files = [
                "fakespotter_epoch_20_train_acc_95_val_acc_85_auc_92.pth",
                "fakespotter_epoch_16_train_acc_95_val_acc_85_auc_92.pth",
                "fakespotter_epoch_6_train_acc_90_val_acc_85_auc_87.pth"
            ]
        else:
            model_files = [
                "fakespotter_epoch_20_train_acc_95_val_acc_85_auc_92.pth"
            ]

        results = []

        # Process models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, model_file in enumerate(model_files, start=1):
                model_path = MODEL_DIR / model_file
                futures.append(
                    executor.submit(process_single_model, idx, str(model_path), str(video_path))
                )

            for future in concurrent.futures.as_completed(futures):
                model_idx, model_result = future.result()
                model_result["model_index"] = model_idx
                results.append(model_result)

        # Add cleanup task to background
        background_tasks.add_task(cleanup_video, video_path)

        return {"status": "success", "results": results}

    except Exception as e:
        if video_path.exists():
            try:
                video_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_video(video_path):
    """Clean up the video file after processing"""
    try:
        if Path(video_path).exists():
            Path(video_path).unlink()
            print(f"Cleaned up video: {video_path}")
    except Exception as e:
        print(f"Error cleaning up video {video_path}: {e}")
    