import cv2
import os
import shutil
import numpy as np
import glob

SQUARE_SIZE = 0.055 

#frame extraction
def extract_frames(video_path, out_folder, step=10):
    # clear old frames first
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Auto-compute step to get ~40 frames spread across full video
    step = max(step, total // 40)
    print(f"Using step={step} to extract ~40 frames from {total} total")
    
    i, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            cv2.imwrite(f"{out_folder}/frame_{saved:04d}.jpg", frame)
            saved += 1
        i += 1
    cap.release()
    print(f"{out_folder}: {saved} frames extracted")

# extract_frames('../data/calibration/checkerboard_b_1.MP4', '../data/calibration/cam1')
# extract_frames('../data/calibration/checkerboard_b_2.MP4', '../data/calibration/cam2')
# extract_frames('../data/calibration/checkerboard_b_3.MP4', '../data/calibration/cam3')

