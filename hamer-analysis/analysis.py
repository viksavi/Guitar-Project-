import pickle
import numpy as np
import argparse
from collections import defaultdict

MANO_JOINT_NAMES = [
    'Wrist',       'Thumb_CMC',  'Thumb_MCP',  'Thumb_IP',   'Thumb_Tip',
    'Index_MCP',   'Index_PIP',  'Index_DIP',  'Index_Tip',
    'Middle_MCP',  'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP',    'Ring_PIP',   'Ring_DIP',   'Ring_Tip',
    'Pinky_MCP',   'Pinky_PIP',  'Pinky_DIP',  'Pinky_Tip',
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MCP_INDICES       = [1, 5, 9, 13, 17]

parser = argparse.ArgumentParser()
parser.add_argument('--pkl',   type=str, nargs='+', required=True,
                    help='2 or 3 PKL files: --pkl view1.pkl view2.pkl [view3.pkl]')
parser.add_argument('--calib', type=str, required=True)
parser.add_argument('--out',   type=str, default='best_hands.pkl')
args = parser.parse_args()

assert 2 <= len(args.pkl) <= 3, "Pass 2 or 3 pkl files"
assert len(set(args.pkl)) == len(args.pkl), f"Duplicate pkl paths: {args.pkl}"

# ── Calibration ───────────────────────────────────────────────────────────────
with open(args.calib, 'rb') as f:
    calib = pickle.load(f)

VIEW_TO_CAM = {'view1': 'cam1', 'view2': 'cam2', 'view3': 'cam3'}

def joints_to_cam2(joints_3d, view_name):
    """Transform joints FROM this camera's space → cam2 world space."""
    cam = VIEW_TO_CAM[view_name]
    R   = calib[cam]['R']   # cam2 → cam
    T   = calib[cam]['T']   # cam2 → cam
    return (R.T @ (joints_3d.T - T)).T  # (21, 3)

def frame_number(image_name):
    return int(image_name.split('_')[1].split('.')[0])

def hand_score(entry):
    if entry is None:
        return -1.0

    # --- PHYSICAL FEASIBILITY CHECK ---
    depth = entry['pred_cam_full'][2]
    
    # In HaMeR units for this specific video, normal hands are 20-30.
    # Anything > 45.0 is a tiny-bounding-box guitar occlusion glitch!
    if depth > 45.0 or depth < 5.0:
        return 0.0
    # ----------------------------------

    conf    = np.array(entry['confidence'])
    vitpose = entry.get('vitpose_score', 0.0)
    
    if isinstance(vitpose, (list, np.ndarray)):
        vitpose = float(np.mean(vitpose))
    if vitpose < 0.5:
        return 0.0
        
    key_conf = np.mean(conf[FINGERTIP_INDICES + MCP_INDICES])
    return 0.3 * vitpose + 0.7 * key_conf

def pack_hand(entry, view_name):
    """Transform both joints fields to cam2 and return a clean dict."""
    cam = VIEW_TO_CAM[view_name]
    R = calib[cam]['R']  
    T = calib[cam]['T']

    joints_local = np.array(entry['pred_joints_3d'])
    cam_trans    = np.array(entry['pred_cam_full'])
    
    # --- METRIC NORMALIZATION FIX ---
    # HaMeR's baseline depth is ~25.0. True physical depth is ~1.0m.
    metric_scale = 1.0 / 25.0 
    cam_trans_metric = cam_trans * metric_scale
    # -------------------------------------
    
    # Add the METRIC camera translation to the local joints
    joints_cam   = joints_local + cam_trans_metric 
    
    # Transform to world space
    joints_world = joints_to_cam2(joints_cam, view_name)
    
    return {
        'joints3d':         joints_local, 
        'joints3d_world':   joints_world,
        'cam_R':            R,             
        'pred_cam_full':    cam_trans_metric, # Save the corrected metric translation
        'confidence':       np.array(entry['confidence']),
        'vitpose_score':    entry.get('vitpose_score'),
        'image_name':       entry['image_name'],
        'pred_mano_params': entry['pred_mano_params'],
        'best_score':       hand_score(entry),
    }

def load_and_index(pkl_path):
    """Index entries as {frame_number: {'left': entry, 'right': entry}}."""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    index = {}
    for entry in results:
        fn   = frame_number(entry['image_name'])
        side = 'right' if entry['is_right'] else 'left'
        if fn not in index:
            index[fn] = {}
        if side not in index[fn] or hand_score(entry) > hand_score(index[fn][side]):
            index[fn][side] = entry
    return index

# ── Load all views ────────────────────────────────────────────────────────────
print(f"\nLoading {len(args.pkl)} views...")
views = {}
for i, pkl_path in enumerate(args.pkl):
    view_name = f'view{i+1}'
    views[view_name] = load_and_index(pkl_path)
    print(f"  {view_name}: {len(views[view_name])} frames  ← {pkl_path}")

all_frames = sorted(set(fn for v in views.values() for fn in v.keys()))
print(f"\nTotal unique frames: {len(all_frames)}")

# ── Select best view (by LEFT hand score), store both hands ──────────────────
output      = []
missing     = 0
view_counts = defaultdict(int)

for fn in all_frames:

    # --- pick best view based on LEFT hand score ---
    best_view, best_sc = None, -1.0
    for view_name, view_index in views.items():
        left_entry = view_index.get(fn, {}).get('left')
        sc         = hand_score(left_entry)
        if sc > best_sc:
            best_sc, best_view = sc, view_name

    if best_view is None or best_sc <= 0.0:
        missing += 1
        continue

    chosen = views[best_view].get(fn, {})

    left_entry  = chosen.get('left')
    right_entry = chosen.get('right')

    # need at least the left hand (the selection criterion)
    if left_entry is None:
        missing += 1
        continue

    left_data  = pack_hand(left_entry,  best_view)
    right_data = pack_hand(right_entry, best_view) if right_entry is not None else None

    # use left hand image_name as the frame label
    output.append({
        'frame_number': fn,
        'best_view':    best_view,
        'best_score':   best_sc,          # left hand score (selection criterion)
        'image_name':   left_data['image_name'],
        'left':         left_data,
        'right':        right_data,       # None if not detected in this view
    })
    view_counts[best_view] += 1

output.sort(key=lambda x: x['frame_number'])

# ── Save ──────────────────────────────────────────────────────────────────────
with open(args.out, 'wb') as f:
    pickle.dump(output, f)

# ── Summary ───────────────────────────────────────────────────────────────────
n_both  = sum(1 for e in output if e['right'] is not None)
n_left  = len(output) - n_both

print(f"\n{'─'*50}")
print(f"Saved to              : {args.out}")
print(f"Frames with left hand : {len(output)}  (selection criterion)")
print(f"  both hands present  : {n_both}")
print(f"  left only           : {n_left}")
print(f"Frames with no detect : {missing}")
print(f"\nBest-view breakdown (by left hand score):")
for vname, count in sorted(view_counts.items()):
    pct = 100 * count / len(output) if output else 0
    print(f"  {vname}: {count} frames ({pct:.1f}%)")

print(f"\nFirst 10 frames preview:")
for entry in output[:10]:
    l = entry['left']
    r = entry['right']
    l_wrist = l['joints3d_world'][0].round(4)
    r_wrist = r['joints3d_world'][0].round(4) if r else None
    print(f"  frame {entry['frame_number']:>5} | "
          f"view={entry['best_view']} | "
          f"left_score={l['best_score']:.3f} | "
          f"left_wrist_world={l_wrist} | "
          f"right_wrist_world={r_wrist}")