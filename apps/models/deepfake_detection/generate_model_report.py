from pathlib import Path
import os

from utils import generate_summary_report, prepare_video_paths
from degraded_fake_spotter import FakeSpotter

BASE_PATH = Path(__file__).resolve().parent
MODEL_DIR = BASE_PATH / "model_checkpoints"
REPORT_PATH = BASE_PATH / "model_evaluation_report"
DATA_DIR = BASE_PATH / "data"


print("\n" + "="*50)
print("Generating Evaluation Report")
print("="*50 + "\n")
        
for item in (os.listdir(MODEL_DIR)):
    
    #load model
    detector = FakeSpotter()
    if not os.path.exists(os.path.join(MODEL_DIR,item)):
        raise FileNotFoundError(f"Model not found: {os.path.join(MODEL_DIR,item)}")
    detector.load_model(str(os.path.join(MODEL_DIR,item)))
    
    train_videos, train_labels, val_videos, val_labels = prepare_video_paths(DATA_DIR)
    
    #evaluation report
    eval_results = generate_summary_report(detector, val_videos, val_labels, output_file=str(REPORT_PATH/item[:-4])+'.txt')