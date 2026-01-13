import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm


FRAME_SKIP = 10
IMG_SIZE = 224
FRAMES_PER_VIDEO = 5


DATA_DIR = "/content/drive/MyDrive/DeepFake Detection/FaceForensicsData"
REAL_VIDEOS = os.path.join(DATA_DIR, "original_sequences/youtube/c23/videos")
FAKE_VIDEOS = os.path.join(DATA_DIR, "manipulated_sequences/Deepfakes/c23/videos")


OUTPUT_DIR = "/content/drive/MyDrive/DeepFake Detection/dataset_resnet"
os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)


detector = MTCNN()


def extract_faces_from_video(video_path, output_folder, label_prefix):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved_count >= FRAMES_PER_VIDEO:
            break
        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                face = rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                filename = f"{label_prefix}_{saved_count}.jpg"
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                saved_count += 1
        frame_count += 1
    cap.release()


for video_file in tqdm(os.listdir(REAL_VIDEOS), desc="Processing Real Videos"):
    video_path = os.path.join(REAL_VIDEOS, video_file)
    extract_faces_from_video(video_path, os.path.join(OUTPUT_DIR, "real"), video_file.replace(".mp4", ""))


for video_file in tqdm(os.listdir(FAKE_VIDEOS), desc="Processing Fake Videos"):
    video_path = os.path.join(FAKE_VIDEOS, video_file)
    extract_faces_from_video(video_path, os.path.join(OUTPUT_DIR, "fake"), video_file.replace(".mp4", ""))
