import pickle
import numpy as np
import argparse
from collections import defaultdict

# joint constants
MANO_JOINT_NAMES = [
    'Wrist',       'Thumb_CMC',  'Thumb_MCP',  'Thumb_IP',   'Thumb_Tip',
    'Index_MCP',   'Index_PIP',  'Index_DIP',  'Index_Tip',
    'Middle_MCP',  'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP',    'Ring_PIP',   'Ring_DIP',   'Ring_Tip',
    'Pinky_MCP',   'Pinky_PIP',  'Pinky_DIP',  'Pinky_Tip',
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MCP_INDICES       = [1, 5, 9,  13, 17]

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', type=str, nargs='+', required=True,
                    help='2 or 3 PKL files: --pkl view1.pkl view2.pkl [view3.pkl]')
parser.add_argument('--calib', type=str, required=True,
                    help='Path to calibration.pkl')
parser.add_argument('--out', type=str, default='best_hands.pkl')
args = parser.parse_args()

assert 2 <= len(args.pkl) <= 3, "Pass 2 or 3 pkl files"

# calibration loading
with open(args.calib, 'rb') as f:
    calib = pickle.load(f)

VIEW_TO_CAM = {'view1': 'cam1', 'view2': 'cam2', 'view3': 'cam3'}

def joints_to_cam2(joints_3d, view_name):
    """Transform joints from any camera space → cam2 world space."""
    cam = VIEW_TO_CAM[view_name]
    R   = calib[cam]['R']
    T   = calib[cam]['T']
    return (R @ joints_3d.T + T).T  # (21, 3)

def frame_number(image_name):
    return int(image_name.split('_')[1].split('.')[0])

# defining score for selecting best-view
def hand_score(entry):
    if entry is None:
        return -1.0
    conf    = np.array(entry['confidence'])
    vitpose = entry.get('vitpose_score', 0.0)
    if isinstance(vitpose, (list, np.ndarray)):
        vitpose = float(np.mean(vitpose))
    if vitpose < 0.5:
        return 0.0
    key_conf = np.mean(conf[FINGERTIP_INDICES + MCP_INDICES])
    return 0.3 * vitpose + 0.7 * key_conf

def load_and_index(pkl_path):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    index = {}
    for entry in results:
        if not entry['is_right']:
            continue
        fn = frame_number(entry['image_name'])
        if fn not in index or hand_score(entry) > hand_score(index[fn]):
            index[fn] = entry
    return index

# ── Load all views ────────────────────────────────────────────────────────────
print(f"\nLoading {len(args.pkl)} views...")
views = {}
for i, pkl_path in enumerate(args.pkl):
    view_name = f'view{i+1}'
    views[view_name] = load_and_index(pkl_path)
    print(f"  {view_name}: {len(views[view_name])} frames  ← {pkl_path}")

all_frames  = sorted(set(fn for v in views.values() for fn in v.keys()))
print(f"\nTotal unique frames: {len(all_frames)}")

# Select best view + transform to cam2 space 
output      = []
missing     = 0
view_counts = defaultdict(int)

for fn in all_frames:
    best_entry, best_view, best_sc = None, None, -1.0

    for view_name, view_index in views.items():
        entry = view_index.get(fn)
        sc    = hand_score(entry)
        if sc > best_sc:
            best_sc, best_entry, best_view = sc, entry, view_name

    if best_entry is None or best_sc <= 0.0:
        missing += 1
        continue

    joints_cam   = np.array(best_entry['pred_joints_3d'])  # original camera space
    joints_world = joints_to_cam2(joints_cam, best_view)   # cam2 world space

    output.append({
        'frame_number':     fn,
        'best_view':        best_view,
        'best_score':       best_sc,
        'joints3d':         joints_cam,          # original camera space
        'joints3d_world':   joints_world,         # cam2 world space
        'pred_cam_full':    np.array(best_entry['pred_cam_full']),
        'confidence':       np.array(best_entry['confidence']),
        'vitpose_score':    best_entry.get('vitpose_score'),
        'image_name':       best_entry['image_name'],
        'pred_mano_params': best_entry['pred_mano_params'],
    })
    view_counts[best_view] += 1

output.sort(key=lambda x: x['frame_number'])

# ── Save ──────────────────────────────────────────────────────────────────────
with open(args.out, 'wb') as f:
    pickle.dump(output, f)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Saved to              : {args.out}")
print(f"Frames with valid hand: {len(output)}")
print(f"Frames with no detect : {missing}")
print(f"\nBest-view breakdown:")
for vname, count in sorted(view_counts.items()):
    pct = 100 * count / len(output) if output else 0
    print(f"  {vname}: {count} frames ({pct:.1f}%)")

print(f"\nFirst 10 frames preview:")
for entry in output[:10]:
    wrist_cam   = entry['joints3d'][0].round(4)
    wrist_world = entry['joints3d_world'][0].round(4)
    print(f"  frame {entry['frame_number']:>5} | "
          f"view={entry['best_view']} | "
          f"score={entry['best_score']:.3f} | "
          f"wrist_cam={wrist_cam} → wrist_world={wrist_world}")