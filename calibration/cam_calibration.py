import cv2
import os
import shutil
import numpy as np
import glob
import pickle
import cv2.aruco as aruco

# ChArUco board parameters — count squares, not inner corners
SQUARES_X   = 7   # count the columns of squares
SQUARES_Y   = 5   # count the rows of squares
SQUARE_SIZE = 0.055
MARKER_SIZE = SQUARE_SIZE * 0.8  # ArUco marker is ~80% of square size

# Define the board
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
charuco_board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE,
    MARKER_SIZE,
    aruco_dict
)
detector = aruco.CharucoDetector(charuco_board)


#frame extraction by timestamps to make it synchronized
def detect_corners_charuco(img_folder):
    obj_points, img_points = [], []
    img_size = None
    paths    = sorted(glob.glob(f"{img_folder}/*.jpg"))

    for path in paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= 6:
            obj_pts, img_pts = charuco_board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            if obj_pts is not None:
                obj_points.append(obj_pts)
                img_points.append(img_pts)

    print(f"{img_folder}: {len(obj_points)}/{len(paths)} frames detected")
    return obj_points, img_points, img_size

# First get min duration across all videos
def get_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total / fps

videos = [
    '../data/calibration/checkerboard_b_1.MP4',
    '../data/calibration/checkerboard_b_2.MP4',
    '../data/calibration/checkerboard_b_3.MP4',
]

folders = [
    '../data/calibration/cam1',
    '../data/calibration/cam2',
    '../data/calibration/cam3',
]

min_duration = min(get_duration(v) for v in videos)
# print(f"Shortest video: {min_duration:.2f} seconds")

# # Now extract same timestamps for all cameras
# for video, folder in zip(videos, folders):
#     extract_frames_by_time(video, folder, 
#                            duration=min_duration, 
#                            n_frames=40)
    
# detect corners of the checkerboard
def detect_corners(img_folder):
    obj_points, img_points, frame_indices = [], [], []
    img_size = None
    paths = sorted(glob.glob(f"{img_folder}/*.jpg"))

    for idx, path in enumerate(paths):
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= 6:
            obj_pts, img_pts = charuco_board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            if obj_pts is not None and len(obj_pts) >= 6:
                obj_points.append(obj_pts)
                img_points.append(img_pts)
                frame_indices.append(idx)  # ← track frame index

    print(f"{img_folder}: {len(obj_points)}/{len(paths)} frames detected")
    return obj_points, img_points, img_size, frame_indices

# Get the intrinsics (one per camera)
def calibrate_camera(obj_points, img_points, img_size):
    ret, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f"  RMS reprojection error: {ret:.4f}px  ← should be < 1.0")
    print(f"  Focal length : fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  Principal pt : cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    return K, dist

# Cell 4 — extrinsics (camera positions relative to cam2)
def stereo_calibrate(obj_pts_1, img_pts_1, idx_1,
                     obj_pts_2, img_pts_2, idx_2,
                     K1, dist1, K2, dist2, img_size):

    # Find frames detected in BOTH cameras
    common_idx = sorted(set(idx_1) & set(idx_2))
    print(f"  Common frames: {len(common_idx)}")

    map1 = {idx: i for i, idx in enumerate(idx_1)}
    map2 = {idx: i for i, idx in enumerate(idx_2)}

    matched_obj  = []
    matched_img1 = []
    matched_img2 = []

    for idx in common_idx:
        o1 = obj_pts_1[map1[idx]]
        o2 = obj_pts_2[map2[idx]]
        i1 = img_pts_1[map1[idx]]
        i2 = img_pts_2[map2[idx]]

        # Must have same number of corners
        if len(o1) == len(o2):
            matched_obj.append(o1)
            matched_img1.append(i1)
            matched_img2.append(i2)

    print(f"  Matched frames with equal corners: {len(matched_obj)}")

    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        matched_obj, matched_img1, matched_img2,
        K1, dist1, K2, dist2,
        img_size, flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"  Stereo RMS: {ret:.4f}px")
    print(f"  Translation: {T.flatten().round(4)} meters")
    return R, T

obj1, img1, size1, idx1 = detect_corners('../data/calibration/cam1')
obj2, img2, size2, idx2 = detect_corners('../data/calibration/cam2')
obj3, img3, size3, idx3 = detect_corners('../data/calibration/cam3')

print("Camera 1:"); K1, dist1 = calibrate_camera(obj1, img1, size1)
print("Camera 2:"); K2, dist2 = calibrate_camera(obj2, img2, size2)
print("Camera 3:"); K3, dist3 = calibrate_camera(obj3, img3, size3)

# cam2 as reference 
print("Cam2 → Cam1:")
R_2_1, T_2_1 = stereo_calibrate(obj2, img2, idx2, obj1, img1, idx1, K2, dist2, K1, dist1, size2)

print("Cam2 → Cam3:")
R_2_3, T_2_3 = stereo_calibrate(obj2, img2, idx2, obj3, img3, idx3, K2, dist2, K3, dist3, size2)

calibration = {
    'cam1': {'K': K1, 'dist': dist1, 'R': R_2_1, 'T': T_2_1},
    'cam2': {'K': K2, 'dist': dist2, 'R': np.eye(3), 'T': np.zeros((3,1))},
    'cam3': {'K': K3, 'dist': dist3, 'R': R_2_3, 'T': T_2_3},
}

with open('../data/calibration/calibration.pkl', 'wb') as f:
    pickle.dump(calibration, f)
print("Calibration saved!")


