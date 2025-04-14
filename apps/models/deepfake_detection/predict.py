from model import FakeSpotter
import os
import requests

def download_video_requests(url, output_path="video.mp4"):
    try:
        # Send GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get total file size (if available)
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Download the file
        downloaded = 0
        print(f"Downloading video to {output_path}")
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress
                    if file_size > 0:
                        percent = (downloaded / file_size) * 100
                        print(f"Download progress: {percent:.1f}%", end='\r')
        
        print("\nDownload completed!")
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

file_path = r"apps\models\deepfake_detection\video.mp4"

video_url = "https://ik.imagekit.io/okd4z65n6/videoplayback.mp4"
download_video_requests(video_url, file_path)

if(os.path.exists(file_path)):
    detector = FakeSpotter()

    detector.load_model(r"apps\models\deepfake_detection\model_checkpoints\fakespotter_epoch_6_train_acc_88_val_acc_90_auc_93.pth")

    result = detector.predict(file_path)
    
    os.remove(file_path)

    print(result)
    
else:
    print("Video download failed. Please check the URL or your internet connection.")