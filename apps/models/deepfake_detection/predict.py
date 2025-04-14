from model import FakeSpotter


detector = FakeSpotter()

detector.load_model(r"apps\models\deepfake_detection\model_checkpoints\fakespotter_epoch_6_train_acc_88_val_acc_90_auc_93.pth")

result = detector.predict(r"apps\models\deepfake_detection\data\real\01__outside_talking_still_laughing.mp4")

print(result)