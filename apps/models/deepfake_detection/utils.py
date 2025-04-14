import os
import glob
from sklearn.model_selection import train_test_split
import json
import torchvision.transforms as transforms



def prepare_video_paths(data_dir, test_size=0.2, seed=42):
    # Collect real and fake video file paths
    real_videos = glob.glob(os.path.join(data_dir, "real", "*.mp4"))
    fake_videos = glob.glob(os.path.join(data_dir, "fake", "*.mp4"))

    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")

    # Labels: 0 = real, 1 = fake
    all_videos = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)

    # Split into training and validation sets
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        all_videos, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    return train_videos, train_labels, val_videos, val_labels

def save_dataset_lists(train_videos, train_labels, val_videos, val_labels, output_dir="dataset_lists"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train_videos.json"), "w") as f:
        json.dump(train_videos, f)
    with open(os.path.join(output_dir, "train_labels.json"), "w") as f:
        json.dump(train_labels, f)

    with open(os.path.join(output_dir, "val_videos.json"), "w") as f:
        json.dump(val_videos, f)
    with open(os.path.join(output_dir, "val_labels.json"), "w") as f:
        json.dump(val_labels, f)

    print(f"Saved dataset lists to: {output_dir}")


def generate_summary_report(detector, val_videos, val_labels, output_file="evaluation_report.txt"):
    """Generate a summary report of model performance on validation set"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    # Process each validation video
    for i, (video_path, label) in enumerate(zip(val_videos, val_labels)):
        print(f"Processing validation video {i+1}/{len(val_videos)}")
        
        # Get prediction
        result = detector.predict(video_path)
        
        # Store results
        all_preds.append(1 if result['is_fake'] else 0)
        all_probs.append(result['confidence'])
        all_labels.append(label)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Write report
    with open(output_file, "w") as f:
        f.write("FakeSpotter Evaluation Report\n")
        f.write("===========================\n\n")
        f.write(f"Number of validation videos: {len(val_videos)}\n")
        f.write(f"Real videos: {np.sum(np.array(val_labels) == 0)}\n")
        f.write(f"Fake videos: {np.sum(np.array(val_labels) == 1)}\n\n")
        
        f.write("Classification Report:\n")
        f.write("---------------------\n")
        f.write(report)
        f.write("\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("----------------\n")
        f.write("                  Predicted\n")
        f.write("                Real    Fake\n")
        f.write(f"Actual Real     {cm[0][0]:5d}    {cm[0][1]:5d}\n")
        f.write(f"       Fake     {cm[1][0]:5d}    {cm[1][1]:5d}\n\n")
        
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
    
    print(f"Evaluation report saved to {output_file}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    
    return {
        'accuracy': (all_preds == all_labels).mean(),
        'auc': roc_auc,
        'report': report,
        'confusion_matrix': cm
    }


# Data augmentation transformations for more robust training
def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])