import pickle
import numpy as np
import argparse
from collections import defaultdict

MANO_JOINT_NAMES = [
    'Wrist',                        # 0  ← root joint
    'Thumb_CMC',                    # 1
    'Thumb_MCP',                    # 2
    'Thumb_IP',                     # 3
    'Thumb_Tip',                    # 4  ← fingertip
    'Index_MCP',                    # 5
    'Index_PIP',                    # 6
    'Index_DIP',                    # 7
    'Index_Tip',                    # 8  ← fingertip
    'Middle_MCP',                   # 9
    'Middle_PIP',                   # 10
    'Middle_DIP',                   # 11
    'Middle_Tip',                   # 12 ← fingertip
    'Ring_MCP',                     # 13
    'Ring_PIP',                     # 14
    'Ring_DIP',                     # 15
    'Ring_Tip',                     # 16 ← fingertip
    'Pinky_MCP',                    # 17
    'Pinky_PIP',                    # 18
    'Pinky_DIP',                    # 19
    'Pinky_Tip',                    # 20 ← fingertip
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MCP_INDICES       = [1, 5, 9,  13, 17]  # knuckles
PIP_INDICES       = [2, 6, 10, 14, 18]  # middle joints
DIP_INDICES       = [3, 7, 11, 15, 19]  # upper joints

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', type=str, nargs='+', required=True,
                    help='2 or 3 PKL files: --pkl view1.pkl view2.pkl [view3.pkl]')
parser.add_argument('--out', type=str, default='best_hands.pkl')
args = parser.parse_args()

assert 2 <= len(args.pkl) <= 3, "Pass 2 or 3 pkl files"

def frame_number(image_name):
    return int(image_name.split('_')[1].split('.')[0])

def hand_score(entry):
    if entry is None:
        return -1.0
    conf    = np.array(entry['confidence'])
    vitpose = entry.get('vitpose_score', 0.0)
    if isinstance(vitpose, (list, np.ndarray)):
        vitpose = float(np.mean(vitpose))
    if vitpose < 0.5:
        return 0.0
    key_joints = FINGERTIP_INDICES + MCP_INDICES
    key_conf   = np.mean(conf[key_joints])
    return 0.3 * vitpose + 0.7 * key_conf

def load_and_index(pkl_path):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    index = {}
    for entry in results:
        if not entry['is_right']:
            continue                          # skip left hand
        fn = frame_number(entry['image_name'])
        if fn not in index or hand_score(entry) > hand_score(index[fn]):
            index[fn] = entry                 
    return index

print(f"\nLoading {len(args.pkl)} views...")
views = {}
for i, pkl_path in enumerate(args.pkl):
    view_name = f'view{i+1}'
    views[view_name] = load_and_index(pkl_path)
    print(f"  {view_name}: {len(views[view_name])} frames  ← {pkl_path}")

all_frames = sorted(set(fn for v in views.values() for fn in v.keys()))
print(f"\nTotal unique frames: {len(all_frames)}")

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

    output.append({
        'frame_number':  fn,
        'best_view':     best_view,
        'best_score':    best_sc,
        # (21, 3) root-relative in HAMER canonical space
        # wrist = (0,0,0), ready to plug into guitar space later
        'joints3d':      np.array(best_entry['pred_joints_3d']),
        'pred_cam_full': np.array(best_entry['pred_cam_full']),  # keep for later
        'confidence':    np.array(best_entry['confidence']),     # (21,)
        'vitpose_score': best_entry.get('vitpose_score'),
        'image_name':    best_entry['image_name'],
        'pred_mano_params': best_entry['pred_mano_params'],
    })
    view_counts[best_view] += 1

output.sort(key=lambda x: x['frame_number'])

# ── Save ───────────────────────────""" """──────────────────────────────────────────

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
    wrist = entry['joints3d'][0]
    print(f"  frame {entry['frame_number']:>5} | "
          f"view={entry['best_view']} | "
          f"score={entry['best_score']:.3f} | "
          f"wrist={wrist}")