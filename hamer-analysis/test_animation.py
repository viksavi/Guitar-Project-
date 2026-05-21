import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import argparse

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
BONE_COLORS = [
    'red','red','red','red',
    'blue','blue','blue','blue',
    'green','green','green','green',
    'orange','orange','orange','orange',
    'purple','purple','purple','purple',
]

parser = argparse.ArgumentParser()
parser.add_argument('--pkl',    type=str, required=True)
parser.add_argument('--out',    type=str, default='hand_animation.mp4')
parser.add_argument('--fps',    type=int, default=240)
parser.add_argument('--smooth', type=int, default=7)
parser.add_argument('--start',  type=int, default=0)
parser.add_argument('--end',    type=int, default=-1)
parser.add_argument('--space',  type=str, default='world',
                    choices=['world', 'cam'],
                    help='world = cam2 space, cam = original camera space')
args = parser.parse_args()

with open(args.pkl, 'rb') as f:
    output = pickle.load(f)

end    = args.end if args.end != -1 else len(output)
output = output[args.start:end]
print(f"Animating {len(output)} frames")

# Pick coordinate space
key = 'joints3d_world' if args.space == 'world' else 'joints3d'
trajectories = np.array([e[key] for e in output])  # (N, 21, 3)
frame_labels = [e['image_name'] for e in output]
view_labels  = [e['best_view']  for e in output]

# Smooth
if args.smooth > 1:
    w = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
    trajectories = savgol_filter(trajectories, window_length=w, polyorder=2, axis=0)

# Axis limits
pad = 0.02
x_min, x_max = trajectories[:,:,0].min()-pad, trajectories[:,:,0].max()+pad
y_min, y_max = trajectories[:,:,1].min()-pad, trajectories[:,:,1].max()+pad
z_min, z_max = trajectories[:,:,2].min()-pad, trajectories[:,:,2].max()+pad

fig = plt.figure(figsize=(9, 8))
ax  = fig.add_subplot(111, projection='3d')

def update(i):
    ax.cla()
    joints = trajectories[i]

    for (a, b), color in zip(BONES, BONE_COLORS):
        ax.plot(*zip(joints[a], joints[b]), color=color, linewidth=2, alpha=0.8)

    ax.scatter(*joints.T,                  c='gray', s=15, zorder=5)
    ax.scatter(*joints[FINGERTIP_INDICES].T, c='red',  s=60, zorder=6)
    ax.scatter(*joints[0],                 c='black', s=80, marker='s', zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"{frame_labels[i]} | {view_labels[i]} | score={output[i]['best_score']:.0f}")

ani = animation.FuncAnimation(fig, update, frames=len(output), interval=1000//args.fps)
ani.save(args.out, writer='ffmpeg', fps=args.fps, dpi=150)
plt.close()
print(f"Saved → {args.out}")