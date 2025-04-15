import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
import json
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np

from dataset import VideoDataset
from model_architecture import FakeSpotterModel


class FakeSpotter:
    def __init__(self, frame_count=20, img_size=299, batch_size=8):
        """
        Initialize the FakeSpotter model
        
        Args:
            frame_count: Number of frames to extract from each video
            img_size: Input image size for the model
            batch_size: Batch size for training
        """
        self.frame_count = frame_count
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Initialize the model and explicitly move to GPU
        self.model = FakeSpotterModel(frame_count, img_size).to(self.device)
        
        # Ensure model is in float32 precision
        self.model = self.model.to(torch.float32)
        
        # Print model parameters count
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")
        
        # Initialize face detector for prediction
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define transformations - adapted for EfficientNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler()
    
    def train(self, train_videos, train_labels, val_videos, val_labels, epochs=20):
        """
        Train the model
        
        Args:
            train_videos: List of paths to training videos
            train_labels: List of labels (0 for real, 1 for fake)
            val_videos: List of paths to validation videos
            val_labels: List of validation labels
            epochs: Number of training epochs
        """
        # Report dataset sizes
        print(f"Training with {len(train_videos)} videos, validating with {len(val_videos)} videos")
        
        # Create datasets
        train_dataset = VideoDataset(
            train_videos, train_labels, 
            frame_count=self.frame_count, 
            img_size=self.img_size,
            transform=self.transform
        )
        
        val_dataset = VideoDataset(
            val_videos, val_labels, 
            frame_count=self.frame_count, 
            img_size=self.img_size,
            transform=self.transform
        )
        
        num_workers = 0
        print(f"Using {num_workers} worker threads for data loading")
        
        # Create data loaders with appropriate number of workers
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False
        )
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
        # Initialize variables for best model tracking
        best_val_auc = 0.0
        best_epoch = 0
        
        # Target accuracy threshold for stopping training
        target_accuracy = 0.95  # 95%
        
        # Create directory for saving models
        model_save_dir = 'model_checkpoints'
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Track training metrics
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, (frames, labels) in enumerate(loop):
                # Move data to device
                frames = frames.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).view(-1, 1)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                
                # Mixed precision forward pass
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(frames)
                    
                # Ensure correct dtype for loss computation
                outputs = outputs.to(dtype=torch.float32)
                labels = labels.to(dtype=torch.float32)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Track statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                loop.set_postfix(loss=train_loss/(batch_idx+1), acc=100*train_correct/max(1, train_total))
                
                # Monitor GPU memory usage periodically
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # For AUC calculation
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch_idx, (frames, labels) in enumerate(loop):
                    # Move data to device
                    frames = frames.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).view(-1, 1)
                    
                    # Forward pass (no need for autocast in eval mode with no_grad)
                    outputs = self.model(frames)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Collect for AUC
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.cpu().numpy())
                    
                    # Update progress bar
                    loop.set_postfix(loss=val_loss/(batch_idx+1), acc=100*val_correct/max(1, val_total))
            
            # Calculate validation metrics
            val_acc = val_correct / max(1, val_total)
            train_acc = train_correct / max(1, train_total)
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            
            # Update training history
            training_history['train_loss'].append(train_loss/len(train_loader))
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss/len(val_loader))
            training_history['val_acc'].append(val_acc)
            training_history['val_auc'].append(val_auc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Train Acc: {100*train_acc:.2f}%, "
                f"Val Loss: {val_loss/len(val_loader):.4f}, "
                f"Val Acc: {100*val_acc:.2f}%, "
                f"Val AUC: {val_auc:.4f}")
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Update learning rate based on validation performance
            scheduler.step(val_auc)
            
            # Save model for current epoch
            model_save_path = os.path.join(model_save_dir, f'fakespotter_epoch_{epoch+1}_train_acc_{int(train_acc*100)}_val_acc_{int(val_acc*100)}_auc_{int(val_auc*100)}.pth')
            self.save_model(model_save_path)
            print(f"Saved model for epoch {epoch+1} with AUC: {val_auc:.4f}")
            
            # Update best model tracking
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                
                # Save the best model separately
                best_model_path = 'fakespotter_best_model.pth'
                self.save_model(best_model_path)
                print(f"Updated best model with AUC: {val_auc:.4f}")
            
            # Check if both training and validation accuracy have reached target (95%)
            if train_acc >= target_accuracy and val_acc >= target_accuracy:
                print(f"Both training accuracy ({100*train_acc:.2f}%) and validation accuracy ({100*val_acc:.2f}%) "
                    f"have reached the target of {100*target_accuracy:.2f}%. Stopping training.")
                break
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(training_history, f)
        
        # Load the best model
        self.load_model('fakespotter_best_model.pth')
        print(f"Training complete. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
        
        return {
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch
        }
    
    def extract_frames(self, video_path):
        """Extract frames from a video for prediction"""
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
            
            # Normalize and convert to tensor
            face_tensor = self.transform(face_frame)
            
            frames.append(face_tensor)
            count += 1
            frame_idx += sampling_interval
            
        cap.release()
        
        # Pad if needed
        if len(frames) == 0:  # Handle case where no frames were extracted
            # Create a dummy tensor
            dummy_tensor = torch.zeros(3, self.img_size, self.img_size)
            frames = [dummy_tensor] * self.frame_count
        else:
            while len(frames) < self.frame_count:
                # Create a zero tensor
                frames.append(torch.zeros_like(frames[0]))
            
        # Stack frames into a single tensor [seq_len, C, H, W]
        return torch.stack(frames)
    
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
    
    def predict(self, video_path):
        """
        Predict if a video is fake or real
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with prediction results
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Add batch dimension
        frames = frames.unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            pred = self.model(frames).item()
        
        # Generate heatmap of suspicious regions
        heatmap = self._generate_attention_heatmap(frames[0])
        
        return {
            'is_fake': bool(pred > 0.75),
            'confidence': float(pred) if pred > 0.75 else float(1 - pred),
            #'heatmap': heatmap
        }
    
    def _generate_attention_heatmap(self, frames):
        """
        Generate heatmap highlighting potentially manipulated regions using Grad-CAM
        
        Args:
            frames: Video frames tensor
            
        Returns:
            List of heatmaps for each frame
        """
        heatmaps = []
        
        # Save the original model state
        original_mode = self.model.training
        
        try:
            # Set model to training mode temporarily for backward pass
            self.model.train()
            
            # Process each frame
            for frame in frames:
                # Add batch dimension
                frame_batch = frame.unsqueeze(0).to(self.device)
                
                # Forward pass with gradient
                frame_batch.requires_grad_()
                
                # Get the features from EfficientNet (last convolutional layer)
                # We need to register hooks to get both activations and gradients
                activations = []
                gradients = []
                
                def save_activation(module, input, output):
                    activations.append(output)
                    
                def save_gradient(grad):
                    gradients.append(grad)
                
                # Register forward hook on the last conv layer of EfficientNet
                last_conv_layer = self.model.feature_extractor.efficientnet.features[-1]
                forward_hook = last_conv_layer.register_forward_hook(save_activation)
                
                # Forward pass through the model
                # Add sequence dimension to match model's expected input
                output = self.model(frame_batch.unsqueeze(0))
                
                # Get last conv activations
                act = activations[0]
                
                # Register backward hook
                act.register_hook(save_gradient)
                
                # Backward pass for the output we want to explain
                output.backward()
                
                # Remove hook after use
                forward_hook.remove()
                
                # Get gradients
                grads = gradients[0]
                
                # Global average pooling on gradients
                weights = torch.mean(grads, dim=[2, 3])
                
                # Create weighted combination of activation maps
                _, num_channels, height, width = act.size()
                
                # Create an empty heatmap tensor
                cam = torch.zeros(height, width, dtype=torch.float32, device=self.device)
                
                # Weighted sum of activation maps
                for i in range(num_channels):
                    cam += weights[0, i] * act[0, i]
                
                # ReLU on the heatmap
                cam = torch.clamp(cam, min=0)
                
                # Normalize the heatmap
                if torch.max(cam) > 0:
                    cam = cam / torch.max(cam)
                
                # Convert to numpy and resize
                cam = cam.cpu().detach().numpy()
                cam = cv2.resize(cam, (self.img_size, self.img_size))
                
                heatmaps.append(cam)
                
                # Clear gradients
                self.model.zero_grad()
        
        finally:
            # Restore the original model mode (evaluation mode)
            if not original_mode:
                self.model.eval()
        
        return heatmaps
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def generate_visualizations(self, output_dir="visualizations"):
        """Generate visualizations for the model architecture"""
        try:            
            os.makedirs(output_dir, exist_ok=True)
            
            # Try to generate model graph with torchviz
            try:
                # Create dummy input
                dummy_input = torch.zeros(1, self.frame_count, 3, self.img_size, self.img_size).to(self.device)
                
                # Forward pass
                output = self.model(dummy_input)
                
                # Generate model graph
                dot = make_dot(output, params=dict(self.model.named_parameters()))
                dot.render(os.path.join(output_dir, "model_graph"), format="png")
                print("Model graph visualization created successfully")
            except Exception as e:
                print(f"Could not generate model graph visualization: {e}")
                print("To generate model graph, please install Graphviz and ensure it's in your system PATH")
                print("Download from: https://graphviz.org/download/")
            
            # Plot training history if available (this part doesn't require Graphviz)
            if os.path.exists('training_history.json'):
                try:
                    import matplotlib.pyplot as plt
                    
                    with open('training_history.json', 'r') as f:
                        history = json.load(f)
                    
                    # Plot training & validation accuracy
                    plt.figure(figsize=(10, 5))
                    plt.plot(history['train_acc'], label='Training Accuracy')
                    plt.plot(history['val_acc'], label='Validation Accuracy')
                    plt.title('Model Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
                    
                    # Plot training & validation loss
                    plt.figure(figsize=(10, 5))
                    plt.plot(history['train_loss'], label='Training Loss')
                    plt.plot(history['val_loss'], label='Validation Loss')
                    plt.title('Model Loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
                    
                    # Plot AUC
                    plt.figure(figsize=(10, 5))
                    plt.plot(history['val_auc'], label='Validation AUC')
                    plt.title('Model AUC')
                    plt.ylabel('AUC')
                    plt.xlabel('Epoch')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'auc_plot.png'))
                    
                    print(f"Training history plots generated in {output_dir}")
                except Exception as e:
                    print(f"Could not generate training history plots: {e}")
                    print("Make sure matplotlib is installed: pip install matplotlib")
                    
        except ImportError as e:
            print(f"Visualization requires additional packages: {e}")
            print("Install with: pip install torchviz matplotlib")
