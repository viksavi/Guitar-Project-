import pickle
import numpy as np
import cv2
import argparse
import os

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MCP_INDICES       = [1, 5, 9, 13, 17]
BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
BONE_COLORS_BGR = [
    (0,0,255),(0,0,255),(0,0,255),(0,0,255),         # Thumb   red
    (255,0,0),(255,0,0),(255,0,0),(255,0,0),         # Index   blue
    (0,255,0),(0,255,0),(0,255,0),(0,255,0),         # Middle  green
    (0,165,255),(0,165,255),(0,165,255),(0,165,255), # Ring    orange
    (128,0,128),(128,0,128),(128,0,128),(128,0,128), # Pinky   purple
]

parser = argparse.ArgumentParser()
parser.add_argument('--pkl',   type=str, required=True,  help='best_hands.pkl')
parser.add_argument('--video', type=str, required=True,  help='cam2 video file')
parser.add_argument('--calib', type=str, required=True,  help='calibration.pkl')
parser.add_argument('--out',   type=str, default='overlay.mp4')
parser.add_argument('--fps',   type=int, default=240)
args = parser.parse_args()

# ── Load 
with open(args.pkl, 'rb') as f:
    output = pickle.load(f)

with open(args.calib, 'rb') as f:
    calib = pickle.load(f)

# Index output by frame number for fast lookup
frame_index = {e['frame_number']: e for e in output}

# cam2 intrinsics for 2D projection
K2   = calib['cam2']['K']    # (3,3)
dist2= calib['cam2']['dist'] # distortion coefficients

def project_joints_2d(joints_world):
    """Project 3D world joints → 2D image using cam2 intrinsics."""
    # joints already in cam2 space → just apply K
    # rvec=0, tvec=0 since joints are already in cam2 frame
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    pts2d, _ = cv2.projectPoints(
        joints_world.astype(np.float64),
        rvec, tvec, K2, dist2
    )
    return pts2d.reshape(-1, 2)

def draw_hand(frame, joints2d):
    """Draw skeleton on frame."""
    joints2d = joints2d.astype(int)

    # Draw bones
    for (a, b), color in zip(BONES, BONE_COLORS_BGR):
        pt1 = tuple(joints2d[a])
        pt2 = tuple(joints2d[b])
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for i, pt in enumerate(joints2d):
        if i in FINGERTIP_INDICES:
            cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)   # red fingertips
        elif i == 0:
            cv2.circle(frame, tuple(pt), 7, (0, 0, 0),   -1)   # black wrist
        else:
            cv2.circle(frame, tuple(pt), 4, (200, 200, 200), -1)  # gray joints

    return frame

# ── Process video ─────────────────────────────────────────────────────────────
cap     = cv2.VideoCapture(args.video)
W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
out     = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

frame_idx  = 0
hits, miss = 0, 0

print(f"Processing video: {W}x{H} @ {src_fps}fps")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frame_index:
        entry      = frame_index[frame_idx]
        joints_w   = entry['joints3d_world']      # (21,3) in cam2 space
        joints2d   = project_joints_2d(joints_w)  # (21,2) in pixels

        # Draw best view label
        cv2.putText(frame,
                    f"frame {frame_idx} | {entry['best_view']} | score={entry['best_score']:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame = draw_hand(frame, joints2d)
        hits += 1
    else:
        # No detection for this frame
        cv2.putText(frame, f"frame {frame_idx} | no detection",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        miss += 1

    out.write(frame)
    frame_idx += 1

    if frame_idx % 500 == 0:
        print(f"  Processed {frame_idx} frames...")

cap.release()
out.release()
print(f"\nSaved → {args.out}")
print(f"Frames with overlay : {hits}")
print(f"Frames without      : {miss}")