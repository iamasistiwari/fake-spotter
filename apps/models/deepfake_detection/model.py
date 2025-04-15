import os
import torch
from torchvision import transforms
import cv2
import logging
import threading
import queue
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FakeSpotter")

# Global request queue for throttling
request_queue = queue.Queue(maxsize=5)  # Limit concurrent requests
active_workers = threading.Semaphore(2)  # Limit active workers

class LightweightFaceDetector:
    """A more CPU efficient face detector"""
    def _init_(self):
        # Use a more efficient face detector model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Set more conservative parameters for speed
        self.scale_factor = 1.2  # Increase from 1.1 to speedup (each scaling step increases by 20% instead of 10%)
        self.min_neighbors = 3  # Reduce from 5 for faster detection
        self.min_size = (60, 60)  # Increase from (30, 30) to filter small faces
        
    def detect(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces


class FakeSpotter:
    def _init_(self, frame_count=16, img_size=299, batch_size=4, model_path=None):
        """
        Initialize the FakeSpotter model with optimized defaults for Linux VPS
        
        Args:
            frame_count: Reduced number of frames (from 20 to 16) for faster processing
            img_size: Reduced image size (from 299 to 224) for faster processing
            batch_size: Reduced batch size (from 8 to 4) for lower memory usage
            model_path: Optional path to a pre-trained model
        """
        self.frame_count = frame_count
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Force CPU mode for consistent performance on VPS
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set threading optimizations for PyTorch
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)  # Limit CPU threads to avoid oversubscription
        logger.info(f"PyTorch using {torch.get_num_threads()} threads")
        
        # Initialize the model
        from model_architecture import FakeSpotterModel
        self.model = FakeSpotterModel(frame_count, img_size).to(self.device)
        
        # Ensure model is in float32 precision
        self.model = self.model.to(torch.float32)
        
        # Print model parameters count
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # Initialize optimized face detector
        self.face_detector = LightweightFaceDetector()
        
        # Define transformations - adapted for EfficientNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def extract_frames(self, video_path):
        """Extract frames from a video for prediction with optimizations"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            # Return empty frames that will be padded
            return self._create_empty_frames()
        
        # Get total frames and calculate sampling interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error(f"Invalid frame count ({total_frames}) for video: {video_path}")
            cap.release()
            return self._create_empty_frames()
            
        # Get video dimensions early
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Skip videos that are too small - often these are corrupt
        if width < 100 or height < 100:
            logger.warning(f"Video dimensions too small: {width}x{height}")
            cap.release()
            return self._create_empty_frames()
            
        # Calculate optimized frame sampling - equidistant sampling
        sampling_interval = max(1, total_frames // self.frame_count)
        frame_indices = [i * sampling_interval for i in range(min(self.frame_count, total_frames))]
        
        # Process each sampled frame
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face and crop
            face_frame = self._detect_and_crop_face(frame)
            
            # Apply transformations
            face_tensor = self.transform(face_frame)
            frames.append(face_tensor)
            
        cap.release()
        
        # Pad frames if needed
        return self._pad_frames(frames)
    
    def _create_empty_frames(self):
        """Create empty frames for error cases"""
        dummy_tensor = torch.zeros(3, self.img_size, self.img_size)
        return torch.stack([dummy_tensor] * self.frame_count)
    
    def _pad_frames(self, frames):
        """Pad frames to required length"""
        if not frames:
            return self._create_empty_frames()
            
        # If we have some frames but not enough, duplicate the last frame
        if len(frames) < self.frame_count:
            last_frame = frames[-1] if frames else torch.zeros(3, self.img_size, self.img_size)
            frames.extend([last_frame] * (self.frame_count - len(frames)))
        
        # If we have too many frames, take the first self.frame_count
        frames = frames[:self.frame_count]
            
        # Stack frames into a single tensor [seq_len, C, H, W]
        return torch.stack(frames)
    
    def _detect_and_crop_face(self, frame):
        """Detect and crop the face from a frame with optimizations"""
        # Resize frame first for faster face detection
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        
        if scale < 1.0:
            frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            frame_resized = frame
        
        # Detect faces
        faces = self.face_detector.detect(frame_resized)
        
        # If no face is detected, return the resized full frame
        if len(faces) == 0:
            return cv2.resize(frame, (self.img_size, self.img_size))
        
        # Get the largest face (by area)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Scale coordinates back to original image size
        if scale < 1.0:
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        
        # Add some margin to the face bounding box (15% on each side)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        
        # Calculate coordinates with margin, ensuring they're within image bounds
        height, width = frame.shape[:2]
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(width, x + w + margin_x)
        end_y = min(height, y + h + margin_y)
        
        # Crop the face region
        face_region = frame[start_y:end_y, start_x:end_x]
        
        # Resize to the target size
        return cv2.resize(face_region, (self.img_size, self.img_size))
    
    def predict(self, video_path):
        """
        Predict if a video is fake or real with performance monitoring
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Limit concurrent processing
            with active_workers:
                logger.info(f"Starting prediction for {video_path}")
                
                # Extract frames (most CPU intensive operation)
                logger.info(f"Extracting frames from {video_path}")
                frame_start = time.time()
                frames = self.extract_frames(video_path)
                logger.info(f"Frame extraction took {time.time() - frame_start:.2f}s")
                
                # Add batch dimension
                frames = frames.unsqueeze(0).to(self.device)
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Make prediction
                inference_start = time.time()
                with torch.no_grad():
                    pred = self.model(frames).item()
                logger.info(f"Model inference took {time.time() - inference_start:.2f}s")
                
                # Format result
                is_fake = bool(pred > 0.75)
                confidence = float(pred) if pred > 0.75 else float(1 - pred)
                
                processing_time = time.time() - start_time
                logger.info(f"Prediction completed in {processing_time:.2f}s: is_fake={is_fake}, confidence={confidence:.4f}")
                
                return {
                    'is_fake': is_fake,
                    'confidence': confidence,
                    'processing_time': processing_time
                }
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': str(e),
                'is_fake': None,
                'confidence': 0.0
            }
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model with error handling"""
        try:
            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
                
            # Load with map_location to ensure it loads on CPU
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode immediately
            logger.info(f"Model loaded from {path}")
            
            # Run a quick inference test with dummy data
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.frame_count, 3, self.img_size, self.img_size).to(self.device)
                _ = self.model(dummy_input)
                
            logger.info("Model test inference successful")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise