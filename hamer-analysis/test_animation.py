import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse


# ── MANO ─────────────────────────────────────────────────────────────────────
from smplx import MANO 

parser = argparse.ArgumentParser()
parser.add_argument('--hands',      type=str, required=True)
parser.add_argument('--mano_dir',   type=str, required=True,
                    help='Path to MANO model dir containing MANO_RIGHT.pkl')
parser.add_argument('--out',        type=str, default='hand_mesh.gif')
parser.add_argument('--fps',        type=int, default=15)
parser.add_argument('--step',       type=int, default=8)
parser.add_argument('--max_frames', type=int, default=100)
args = parser.parse_args()

# ── Load MANO ─────────────────────────────────────────────────────────────────
# Replace this:
mano = MANO(args.mano_dir, use_pca=False, is_rhand=True)
mano.eval()
faces = mano.faces

# With this — point to the folder, not the file:
mano = MANO(
    model_path=args.mano_dir,   # folder containing MANO_RIGHT.pkl
    use_pca=False,
    is_rhand=True,
    flat_hand_mean=True,
)
mano.eval()
faces = mano.faces.astype(int)  # (1538, 3)

# ── Load hand data ────────────────────────────────────────────────────────────
with open(args.hands, 'rb') as f:
    best_hands = pickle.load(f)

frames_data = best_hands[::args.step][:args.max_frames]
print(f"Total in pkl  : {len(best_hands)}")
print(f"Animating     : {len(frames_data)} frames")

# ── Reconstruct vertices per frame ────────────────────────────────────────────
from scipy.spatial.transform import Rotation as R

def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (..., 3, 3) → axis-angle (..., 3)."""
    orig_shape = rotmat.shape[:-2]
    aa = R.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(*orig_shape, 3)

def get_vertices(entry):
    params = entry['pred_mano_params']

    global_orient_aa = rotmat_to_aa(np.array(params['global_orient']))  # (1, 3)
    hand_pose_aa     = rotmat_to_aa(np.array(params['hand_pose']))      # (15, 3)

    # Fix: both must be 2D (batch, dims)
    global_orient_t = torch.tensor(global_orient_aa.reshape(1, 3),  dtype=torch.float32)  # (1, 3)
    hand_pose_t     = torch.tensor(hand_pose_aa.reshape(1, 45),     dtype=torch.float32)  # (1, 45)
    betas_t         = torch.tensor(np.array(params['betas']).reshape(1, 10), dtype=torch.float32)  # (1, 10)

    with torch.no_grad():
        out = mano(
            global_orient=global_orient_t,
            hand_pose=hand_pose_t,
            betas=betas_t,
            return_tips=True
        )

    return out.vertices[0].numpy()   # (778, 3)

# ── Pre-compute all vertices + axis limits ────────────────────────────────────
print("Reconstructing vertices from MANO params...")
all_verts = []
for i, entry in enumerate(frames_data):
    verts = get_vertices(entry)
    all_verts.append(verts)
    if i % 20 == 0:
        print(f"  {i}/{len(frames_data)}")

all_verts_np = np.concatenate(all_verts, axis=0)
center = all_verts_np.mean(axis=0)
spread = max(
    all_verts_np[:,0].max() - all_verts_np[:,0].min(),
    all_verts_np[:,1].max() - all_verts_np[:,1].min(),
    all_verts_np[:,2].max() - all_verts_np[:,2].min(),
) / 2 + 0.02

print("Vertices ready.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 8), facecolor='black')
ax  = fig.add_subplot(111, projection='3d', facecolor='black')

ax.set_xlim(center[0] - spread, center[0] + spread)
ax.set_ylim(center[1] - spread, center[1] + spread)
ax.set_zlim(center[2] - spread, center[2] + spread)
ax.set_axis_off()

title = ax.set_title('', color='white', fontsize=9)

# Mesh collection — skin tone color
mesh_collection = Poly3DCollection(
    [],
    alpha=0.85,
    facecolor='#C68642',   # skin tone
    edgecolor='none',
)
ax.add_collection3d(mesh_collection)

# ── Update ────────────────────────────────────────────────────────────────────
def update(frame_idx):
    entry = frames_data[frame_idx]
    verts = all_verts[frame_idx]          # (778, 3)

    triangles = verts[faces]              # (1538, 3, 3)
    mesh_collection.set_verts(triangles)

    title.set_text(
        f"frame {entry['frame_number']:05d}  |  "
        f"view={entry['best_view']}  |  "
        f"score={entry['best_score']:.3f}"
    )
    return [mesh_collection, title]

# ── Animate ───────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    frames=len(frames_data),
    interval=1000 // args.fps,
    blit=False
)

print(f"Saving → {args.out} ...")
if args.out.endswith('.gif'):
    ani.save(args.out, writer='pillow', fps=args.fps,
             savefig_kwargs={'facecolor': 'black'})
else:
    ani.save(args.out, writer='ffmpeg', fps=args.fps,
             savefig_kwargs={'facecolor': 'black'})

print("Done!")
