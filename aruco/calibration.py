import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (6, 4)
SQUARE_SIZE = 55.0  # Millimeters
IMAGE_DIR = "calibration_frames" 
IMAGE_EXTENSION = "*.jpg"     

# Fisheye calibration criteria and flags
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Get list of all images in the directory
search_path = os.path.join(IMAGE_DIR, IMAGE_EXTENSION)
images = glob.glob(search_path)

if not images:
    print(f"Error: No images found matching pattern '{search_path}'")
    exit()

print(f"Found {len(images)} images for calibration.")

# Variable to hold image size (needed for calibration later)
gray_shape = None

# --- PROCESS IMAGE DIRECTORY ---
for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]  # Save width, height
    
    ret_corners, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD_SIZE, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_corners:
        # Instead of manual keystroke 'c', we automatically use every valid image
        refined_corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        objpoints.append(objp)
        imgpoints.append(refined_corners.reshape(1, -1, 2))
        
        # Optional: Display the success frames to visually verify
        display_frame = frame.copy()
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, refined_corners, ret_corners)
        cv2.imshow('Processing Frames', display_frame)
        cv2.waitKey(100) # Briefly pause to see the visualization
    else:
        print(f"Chessboard not found in: {img_path}")

cv2.destroyAllWindows()

# --- FISHEYE CALIBRATION ---
N_OK = len(objpoints)
if N_OK > 15:
    print(f"\nCalculating GoPro Fisheye Calibration using {N_OK} valid frames...")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
    
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray_shape,  # Used the stored shape
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print("\n--- GOPRO CALIBRATION RESULTS ---")
    print(f"RMS Error: {rms}")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nFisheye Distortion Coefficients (D):")
    print(D)
    
    np.savez("gopro_calibration.npz", mtx=K, dist=D)
    print("\nCalibration data saved to 'gopro_calibration.npz'")
else:
    print(f"Not enough valid frames. Found chessboard in only {N_OK} frames (15+ required).")