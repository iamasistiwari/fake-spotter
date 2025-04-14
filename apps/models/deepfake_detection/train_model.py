import os
import numpy as np
import torch

from utils import prepare_video_paths, save_dataset_lists, generate_summary_report
from model import FakeSpotter



# Example usage
if __name__ == "__main__":
    # Verify CUDA is properly installed and configured
    if torch.cuda.is_available():
        print(f"GPU acceleration enabled! Using {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Show additional GPU information
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"Total GPU memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"CUDA capability: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"Number of SMs: {gpu_properties.multi_processor_count}")
        
        # Test CUDA with a simple operation to ensure it's working
        test_tensor = torch.ones(1).cuda()
        print(f"Test tensor device: {test_tensor.device}")
    else:
        print("No GPU found. Running on CPU.")
        print("For optimal performance, consider a machine with a CUDA-compatible GPU.")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize FakeSpotter with optimal batch size based on GPU memory
    if torch.cuda.is_available():
        # Calculate approximate memory needed per sample
        # This is an estimation and may need adjustment
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_mem_gb > 10:  # High-end GPU (>10GB)
            batch_size = 16
        elif gpu_mem_gb > 6:  # Mid-range GPU (6-10GB)
            batch_size = 8
        else:  # Entry-level GPU (<6GB)
            batch_size = 4
    else:
        # CPU mode - smaller batch size to avoid memory issues
        batch_size = 2
        
    print(f"Using batch size: {batch_size}")
    
    # Initialize detector with optimal parameters
    detector = FakeSpotter(
        frame_count=20,  
        img_size=299,
        batch_size=batch_size
    )
    
    
    # Path to data directory
    DATA_DIR = "data"  # Change if needed
    
    # Check if directories exist
    if not os.path.exists(os.path.join(DATA_DIR, "real")) or not os.path.exists(os.path.join(DATA_DIR, "fake")):
        print(f"Error: Could not find 'real' or 'fake' directories in {DATA_DIR}")
        print("Please ensure your data is organized as follows:")
        print("data/")
        print("├── real/")
        print("│   ├── video1.mp4")
        print("│   └── ...")
        print("└── fake/")
        print("    ├── fake1.mp4")
        print("    └── ...")
    else:
        # Prepare dataset
        train_videos, train_labels, val_videos, val_labels = prepare_video_paths(DATA_DIR)
        save_dataset_lists(train_videos, train_labels, val_videos, val_labels)
        
        # Begin training
        print("\n" + "="*50)
        print("Starting FakeSpotter Training Pipeline")
        print("="*50 + "\n")
        
        # Train the model with adaptive epochs based on dataset size
        # For small datasets, we might want more epochs for convergence
        if len(train_videos) < 100:
            epochs = 20
        elif len(train_videos) < 500:
            epochs = 10
        else:
            epochs = 5
            
        print(f"Training for {epochs} epochs based on dataset size")
        
        training_results = detector.train(
            train_videos, 
            train_labels, 
            val_videos, 
            val_labels,
            epochs=epochs
        )
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best validation AUC: {training_results['best_val_auc']:.4f}")
        print(f"Best epoch: {training_results['best_epoch'] + 1}")
        print("="*50 + "\n")
        
        # Save model
        detector.save_model("fakespotter_final_model.pth")
        
        
        # Generate visualizations
        detector.generate_visualizations()
        
        # Generate a comprehensive evaluation report
        print("\n" + "="*50)
        print("Generating Evaluation Report")
        print("="*50 + "\n")
        
        eval_results = generate_summary_report(detector, val_videos, val_labels)
        
        print("\n" + "="*50)
        print("Evaluation Results Summary")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"AUC: {eval_results['auc']:.4f}")
        print("="*50 + "\n")
        
        print("Model training and evaluation complete!")
        print("You can now use the trained model for prediction on new videos.")
        print("\nExample usage:")
        print("detector = FakeSpotter()")
        print("detector.load_model('fakespotter_best_model.pth')")
        print("result = detector.predict('path/to/video.mp4')")
        print("print(f\"Is fake: {result['is_fake']}, Confidence: {result['confidence']:.2f}\")")