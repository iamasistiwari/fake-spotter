import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class VideoDataset(Dataset):
    """Dataset class to load videos and labels"""
    def __init__(self, video_paths, labels, frame_count=20, img_size=299, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.frame_count = frame_count
        self.img_size = img_size
        self.transform = transform
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def __len__(self):
        return len(self.video_paths)
    
    def detect_and_crop_face(self, frame):
        """Detect and crop the face from a frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no face is detected, return the resized full frame
        if len(faces) == 0:
            return cv2.resize(frame, (self.img_size, self.img_size))
        
        # Get the largest face (by area)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
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
    
    def extract_frames(self, video_path):
        """Extract frames from a video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames and calculate sampling interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_interval = max(1, total_frames // self.frame_count)
        
        count = 0
        frame_idx = 0
        
        while count < self.frame_count and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face and crop
            face_frame = self.detect_and_crop_face(frame)
            
            # Apply transformations if available
            if self.transform:
                face_frame = self.transform(face_frame)
            else:
                # Normalize to [-1, 1] range like in TF code
                face_frame = face_frame.astype(np.float32) / 127.5 - 1.0
                # Convert to PyTorch tensor format [C, H, W]
                face_frame = np.transpose(face_frame, (2, 0, 1))
                face_frame = torch.from_numpy(face_frame).float()
            
            frames.append(face_frame)
            count += 1
            frame_idx += sampling_interval
            
        cap.release()
        
        # Pad if needed
        while len(frames) < self.frame_count:
            # Create a zero tensor with the same shape as other frames
            zero_frame = torch.zeros_like(frames[0] if frames else torch.zeros(3, self.img_size, self.img_size))
            frames.append(zero_frame)
            
        # Stack frames into a single tensor [seq_len, C, H, W]
        return torch.stack(frames)
    
    def __getitem__(self, idx):
        """
        Get a single video with its label
        
        Returns:
            frames: Tensor of shape [seq_len, channels, height, width]
            label: Binary label (0 for real, 1 for fake)
        """
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]
            
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Return float32 label explicitly
            return frames, torch.tensor(float(label), dtype=torch.float32)
        except Exception as e:
            print(f"Error processing video {self.video_paths[idx]}: {e}")
            # Return a zero tensor with appropriate shape in case of error
            return torch.zeros(self.frame_count, 3, self.img_size, self.img_size), torch.tensor(float(self.labels[idx]), dtype=torch.float32)
