"""
Outputs:
  - summary_table.png       : Table 5.1 style, fingertips + all joints
  - per_joint_jitter.png    : All 21 joints, WiLoR vs HaMeR
  - depth_comparison.png    : Depth std + jitter per model per video
  - cross_view_spread.png   : Vertex position spread across cam1/2/3
  - analysis_data.json      : All raw numbers
"""

from pathlib import Path
import pickle, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── MANO joints ────────────────────────────────────────────────────────────
JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
    'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
    'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
    'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
    'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip',
]
FINGERTIPS   = [4, 8, 12, 16, 20]
NON_THUMB    = [0, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20]  # all except thumb chain (1,2,3,4)
ALL_JOINTS   = list(range(21))
FINGER_NAMES = ['Thumb','Index','Middle','Ring','Pinky']

COLORS_MODEL = {'wilor': '#4ecdc4', 'hamer': '#ff6b6b'}
COLORS_CAM   = {'cam1': '#ff6b6b', 'cam2': '#4ecdc4', 'cam3': '#ffe66d'}

# LOADERS for reconstruction models data

def load_wilor(pred_dir, K, f_syn, R=None, T=None, limit=-1):
    pred_dir = Path(pred_dir)
    frames_dict = defaultdict(list)
    for p in pred_dir.glob('*_verts.npy'):
        m = re.search(r'frame_(\d+)_(\d+)_verts\.npy', p.name)
        if m:
            frames_dict[int(m.group(1))].append(m.group(2))

    sorted_frames = sorted(frames_dict.keys())
    if limit > 0:
        sorted_frames = sorted_frames[:limit]

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    results = {}

    for fnum in sorted_frames:
        best = {}
        for hidx in frames_dict[fnum]:
            sp = pred_dir / f'frame_{fnum:04d}_{hidx}_score.npy'
            rp = pred_dir / f'frame_{fnum:04d}_{hidx}_is_right.npy'
            if not sp.exists(): continue
            score    = float(np.load(sp).ravel()[0])
            is_right = bool(np.load(rp).ravel()[0])
            if is_right not in best or score > best[is_right][0]:
                best[is_right] = (score, hidx)

        frame_data = {}
        for is_right_raw, (score, hidx) in best.items():
            jp = pred_dir / f'frame_{fnum:04d}_{hidx}_joints3d.npy'
            tp = pred_dir / f'frame_{fnum:04d}_{hidx}_cam_t.npy'
            kp = pred_dir / f'frame_{fnum:04d}_{hidx}_kpts2d.npy'
            if not (jp.exists() and tp.exists() and kp.exists()): continue

            joints_raw = np.load(jp).astype(np.float32)
            cam_t_raw  = np.load(tp).astype(np.float32)
            kpts2d     = np.load(kp).astype(np.float32) # (778, 2) - the 2D pixel coordinates of all mesh vertices 
            is_right   = not is_right_raw

            tz = cam_t_raw[2] * (fx / f_syn) # convert Z to meters using camera focal length
            tx = (kpts2d[0,0] - cx) * tz / fx # back projection - metric X, cx - principal point
            ty = (kpts2d[0,1] - cy) * tz / fy # back projection - metric Y, cy - principal point
            cam_t_metric = np.array([tx, ty, tz], dtype=np.float32)

            joints = joints_raw.copy()
            if not is_right:
                joints[:, 0] *= -1.0 # flip for the left hand
            joints_cam = joints + cam_t_metric
            joints_cam[:, 1] *= -1.0 # Y-up -> Y-down (OpenCV convention)

            if R is not None:
                joints_cam = (R.T @ (joints_cam - T).T).T # Stereo back-projection: (21,3) → subtract T → transpose to (3,21)
                                                          # → rotate with R.T → transpose back to (21,3)

            frame_data[is_right] = joints_cam.astype(np.float32)

        if frame_data:
            results[fnum] = frame_data

    return results, sorted_frames


def load_hamer(pkl_path, K, R=None, T=None, limit=-1):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    frames = defaultdict(list)
    for entry in raw:
        m = re.search(r'frame_(\d+)', entry['image_name'])
        if m:
            frames[int(m.group(1))].append(entry)

    sorted_frames = sorted(frames.keys())
    if limit > 0:
        sorted_frames = sorted_frames[:limit]

    fx = K[0, 0]  # real focal length for metric scaling

    results = {}
    for fnum in sorted_frames:
        best = {}
        for e in frames[fnum]:
            is_right = bool(e['is_right'])
            score = np.mean(e['vitpose_score']) if e.get('vitpose_score') else 0.5
            if is_right not in best or score > best[is_right][0]:
                best[is_right] = (score, e)

        frame_data = {}
        for is_right, (score, e) in best.items():
            joints_local = np.array(e['pred_joints_3d'], dtype=np.float32)  # (21,3) meters
            cam_t        = np.array(e['pred_cam_full'],  dtype=np.float32)  # unnormalized scale
            img_size     = np.array(e['img_size'],       dtype=np.float32)

            # cam_t uses same unnormalized scale as WiLoR cam_t[2] (~18-22)
            # Convert to metric using same formula as WiLoR
            f_syn = 5000.0 / 256.0 * float(img_size.max())
            scale = fx / f_syn
            cam_t_metric = cam_t.copy() * scale  # all 3 axes uniform scale

            # joints_local already in meters — compose with metric translation
            joints_cam = joints_local + cam_t_metric
            joints_cam[:, 1] *= -1.0  # Y-up -> Y-down (OpenCV convention)

            if R is not None:
                joints_cam = (R.T @ (joints_cam - T).T).T

            frame_data[is_right] = joints_cam.astype(np.float32)

        if frame_data:
            results[fnum] = frame_data

    return results, sorted_frames

# METRICS calculation

def compute_jitter_variance(preds, sorted_frames, relative_to_wrist=True):
    """
    Frame-to-frame displacement variance (m/frame)^2 per joint.
    """
    disp_r, disp_l = [], []

    for i in range(1, len(sorted_frames)):
        f0, f1 = sorted_frames[i-1], sorted_frames[i]
        if f0 not in preds or f1 not in preds: continue
        if f1 - f0 > 5: continue # skip large gaps (more than 5 frames)

        for is_right, buf in [(True, disp_r), (False, disp_l)]:
            if is_right not in preds[f0] or is_right not in preds[f1]: continue
            j0 = preds[f0][is_right].copy() 
            j1 = preds[f1][is_right].copy()
            if relative_to_wrist:
                j0 -= j0[0:1] # substract the wrist
                j1 -= j1[0:1]
            buf.append(j1 - j0)   # (21, 3) displacement

    out = {}
    for side, buf in [('right', disp_r), ('left', disp_l)]:
        if not buf:
            out[side] = None; continue
        data = np.array(buf)            # (F, 21, 3)
        mag  = np.linalg.norm(data, axis=2)  # (F, 21)
        pjv  = mag.var(axis=0)          # (21,) variance
        out[side] = {
            'per_joint_var':   pjv,
            'per_joint_std':   mag.std(axis=0),
            'per_joint_mean':  mag.mean(axis=0),
            'all_joints_var':  pjv.mean(),
            'fingertips_var':  pjv[FINGERTIPS].mean(),
            'no_thumb_var':    pjv[NON_THUMB].mean(),
        }
    return out


def compute_depth_stats(preds, sorted_frames):
    """Wrist Z statistics: mean, std (consistency), frame-to-frame jitter."""
    dr, dl = [], []
    for fnum in sorted_frames:
        if fnum not in preds: continue
        if True  in preds[fnum]: dr.append(float(preds[fnum][True][0, 2]))
        if False in preds[fnum]: dl.append(float(preds[fnum][False][0, 2]))

    out = {}
    for side, depths in [('right', dr), ('left', dl)]:
        if len(depths) < 2:
            out[side] = None; continue
        d = np.array(depths)
        jitter = np.abs(np.diff(d))
        out[side] = {
            'mean_m':      float(d.mean()),
            'std_m':       float(d.std()),
            'jitter_mean': float(jitter.mean()),
            'jitter_std':  float(jitter.std()),
            'jitter_p95':  float(np.percentile(jitter, 95)), # ignores the 5% of outliers
            'range_m':     float(d.max() - d.min()),
            'n_frames':    len(depths),
        }
    return out


def compute_cross_view_spread(preds_cam1, preds_cam2, preds_cam3, sorted_frames):
    """
    For each frame where all 3 views detected the same hand:
    compute std of joint positions across the 3 views.
    Returns per-joint spread (std in meters) averaged over frames.
    """
    spread_r, spread_l = [], []
    depth_diff_r, depth_diff_l = [], []

    for fnum in sorted_frames:
        for is_right, spread_buf, depth_buf in [
                (True,  spread_r, depth_diff_r),
                (False, spread_l, depth_diff_l)]:
            # Need all 3 views to have this hand
            if not all(is_right in p.get(fnum, {})
                       for p in [preds_cam1, preds_cam2, preds_cam3]):
                continue

            j1 = preds_cam1[fnum][is_right]  # (21, 3)
            j2 = preds_cam2[fnum][is_right]
            j3 = preds_cam3[fnum][is_right]

            # Stack all 3 views: (3, 21, 3)
            stack = np.stack([j1, j2, j3], axis=0)

            # Per-joint std across views: (21,) in meters
            per_joint_spread = stack.std(axis=0).mean(axis=1)  # (21,)
            spread_buf.append(per_joint_spread)

            # Depth specifically: std of wrist Z across 3 views
            wrist_z = stack[:, 0, 2]   # (3,)
            depth_buf.append(wrist_z.std())

    out = {}
    for side, buf, dbuf in [('right', spread_r, depth_diff_r),
                              ('left',  spread_l, depth_diff_l)]:
        if not buf:
            out[side] = None; continue
        arr   = np.array(buf)   # (F, 21)
        deptharr = np.array(dbuf)
        out[side] = {
            'per_joint_mean_spread': arr.mean(axis=0),   # (21,) avg spread
            'per_joint_max_spread':  arr.max(axis=0),
            'all_joints_mean':       float(arr.mean()),
            'fingertips_mean':       float(arr[:, FINGERTIPS].mean()),
            'no_thumb_mean':         float(arr[:, NON_THUMB].mean()),
            'depth_spread_mean':     float(deptharr.mean()),
            'depth_spread_std':      float(deptharr.std()),
            'n_frames':              len(buf),
        }
    return out

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def _style(ax, title):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#334')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.grid(axis='y', color='#334', linewidth=0.5, alpha=0.6)


def plot_per_joint_jitter(all_jitter, video_labels, out_path):
    """
    Per-joint jitter std, averaged across ALL cams and ALL videos per model.
    One bar group per joint, one bar per model.
    """
    # Aggregate: for each model, average per_joint_std across all cams & videos
    model_avg = {}
    for model in ['wilor', 'hamer']:
        right_arrays, left_arrays = [], []
        for key, jdata in all_jitter.items():
            if not key.startswith(model): continue
            for side, arr_list in [('right', right_arrays), ('left', left_arrays)]:
                if jdata.get(side):
                    arr_list.append(jdata[side]['per_joint_std'])
        model_avg[model] = {
            'right': np.mean(right_arrays, axis=0) * 1000 if right_arrays else None,
            'left':  np.mean(left_arrays,  axis=0) * 1000 if left_arrays  else None,
        }

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.patch.set_facecolor('#0d0d1a')

    x = np.arange(21)
    width = 0.38

    for ax_idx, side in enumerate(['right', 'left']):
        ax = axes[ax_idx]
        _style(ax, f'Per-Joint Jitter Std — {side.capitalize()} Hand (mm/frame) '
                   f'[avg across all cams & videos]')

        for mi, model in enumerate(['wilor', 'hamer']):
            vals = model_avg[model][side]
            if vals is None: continue
            bars = ax.bar(x + mi*width - width/2, vals, width,
                          label=model.upper(),
                          color=COLORS_MODEL[model], alpha=0.85)

        # Mark fingertips
        for ft in FINGERTIPS:
            ax.axvline(ft, color='#ffffff', linewidth=0.8,
                       alpha=0.15, linestyle='--')

        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=45,
                           ha='right', color='#aaaaaa', fontsize=7)
        ax.set_ylabel('Std (mm/frame)', color='#aaaaaa')
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    fig.suptitle('Per-Joint Jitter — WiLoR vs HaMeR\n'
                 '(dashed lines = fingertips, lower = more stable)',
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")


def plot_depth_comparison(all_depth, video_labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d0d1a')

    x = np.arange(len(video_labels))
    width = 0.35

    for ax, (metric_key, title) in zip(axes, [
            ('std_m',        'Within-video depth std (mm)'),
            ('jitter_mean',  'Frame-to-frame depth jitter (mm)')]):
        _style(ax, title)

        for mi, model in enumerate(['wilor', 'hamer']):
            vals = []
            for vid in video_labels:
                vid_key = vid.replace(' ', '_').lower()
                # Average across all 3 cams
                v = _avg_across_cams(all_depth, model, vid_key,
                                     metric_key, metric_key)
                vals.append(v * 1000 if v is not None else 0)

            ax.bar(x + mi*width, vals, width,
                   label=model.upper(),
                   color=COLORS_MODEL[model], alpha=0.85)

        ax.set_xticks(x + width/2)
        ax.set_xticklabels(video_labels, color='#aaaaaa')
        ax.set_ylabel('milimeters', color='#aaaaaa')
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    fig.suptitle('Depth Estimation Consistency — WiLoR vs HaMeR '
                 '(avg across all 3 cams)',
                 color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")

def plot_cross_view_spread(all_spread, video_labels, out_path):
    # Get per-joint spread averaged across videos per model
    model_joint_spread = {}
    for model in ['wilor', 'hamer']:
        right_arrays, left_arrays = [], []
        for vid in video_labels:
            vid_key = vid.replace(' ', '_').lower()
            k = f"{model}_{vid_key}"
            d = all_spread.get(k, {})
            for side, arr_list in [('right', right_arrays), ('left', left_arrays)]:
                if d.get(side):
                    arr_list.append(d[side]['per_joint_mean_spread'])
        model_joint_spread[model] = {
            'right': np.mean(right_arrays, axis=0) if right_arrays else None,
            'left':  np.mean(left_arrays,  axis=0) if left_arrays  else None,
        }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor('#0d0d1a')

    x = np.arange(21)
    width = 0.38

    # Top row: per-joint spread per model
    for ax_idx, side in enumerate(['right', 'left']):
        ax = axes[0, ax_idx]
        _style(ax, f'Cross-View Joint Spread — {side.capitalize()} Hand (m)\n'
                   f'[std of position across cam1/cam2/cam3]')
        for mi, model in enumerate(['wilor', 'hamer']):
            vals = model_joint_spread[model][side]
            if vals is None: continue
            ax.bar(x + mi*width - width/2, vals, width,
                   label=model.upper(),
                   color=COLORS_MODEL[model], alpha=0.85)
        for ft in FINGERTIPS:
            ax.axvline(ft, color='#ffffff', linewidth=0.8,
                       alpha=0.15, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=45,
                           ha='right', color='#aaaaaa', fontsize=7)
        ax.set_ylabel('Spread std (m)', color='#aaaaaa')
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    # Bottom left: overall spread per video
    ax = axes[1, 0]
    _style(ax, 'Overall Cross-View Spread per Video (m)\n'
               '[lower = more consistent across cameras]')
    xv = np.arange(len(video_labels))
    for mi, model in enumerate(['wilor', 'hamer']):
        vals = []
        for vid in video_labels:
            vid_key = vid.replace(' ', '_').lower()
            k = f"{model}_{vid_key}"
            d = all_spread.get(k, {})
            sides = [d.get(s) for s in ['right','left'] if d.get(s)]
            vals.append(np.mean([s['all_joints_mean'] for s in sides])
                        if sides else 0)
        ax.bar(xv + mi*width, vals, width,
               label=model.upper(), color=COLORS_MODEL[model], alpha=0.85)
    ax.set_xticks(xv + width/2)
    ax.set_xticklabels(video_labels, color='#aaaaaa')
    ax.set_ylabel('Mean spread (m)', color='#aaaaaa')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    # Bottom right: depth spread across views
    ax = axes[1, 1]
    _style(ax, 'Wrist Depth Spread Across Views (m)\n'
               '[std of Z across cam1/cam2/cam3 per frame]')
    for mi, model in enumerate(['wilor', 'hamer']):
        vals = []
        for vid in video_labels:
            vid_key = vid.replace(' ', '_').lower()
            k = f"{model}_{vid_key}"
            d = all_spread.get(k, {})
            sides = [d.get(s) for s in ['right','left'] if d.get(s)]
            vals.append(np.mean([s['depth_spread_mean'] for s in sides])
                        if sides else 0)
        ax.bar(xv + mi*width, vals, width,
               label=model.upper(), color=COLORS_MODEL[model], alpha=0.85)
    ax.set_xticks(xv + width/2)
    ax.set_xticklabels(video_labels, color='#aaaaaa')
    ax.set_ylabel('Depth spread std (m)', color='#aaaaaa')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    fig.suptitle('Cross-View Prediction Consistency\n'
                 '(justifies using central camera only for demo)',
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_table(all_jitter, video_labels, out_path):
    metrics = [
        ('Fingertips only',       'fingertips_var'),
        ('All 21 joints',         'all_joints_var'),
        ('No thumb (excl. 1-4)', 'no_thumb_var'),
    ]
    col_labels = ['Model'] + list(video_labels) + ['Mean']

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3.2 * len(metrics)))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle(
        'Fingertip-velocity variance (m/frame)² — relative to wrist\n'
        'averaged across cam1, cam2, cam3',
        color='white', fontsize=12, y=1.01)

    for ax, (subtitle, metric_key) in zip(axes, metrics):
        ax.set_facecolor('#0d0d1a')
        ax.axis('off')
        ax.set_title(subtitle, color='#aef', fontsize=11, style='italic',
                     pad=6, loc='center')

        rows = []
        for model_label, model in [('HaMeR', 'hamer'), ('WiLoR', 'wilor')]:
            row = [model_label]
            vals = []
            for vid in video_labels:
                vid_key = vid.replace(' ', '_').lower()
                v = _avg_across_cams(all_jitter, model, vid_key,
                                     metric_key, metric_key)
                if v is not None:
                    vals.append(v); row.append(f'{v:.3e}')
                else:
                    row.append('N/A')
            mean_v = np.mean(vals) if vals else None
            row.append(f'{mean_v:.3e}' if mean_v else 'N/A')
            rows.append(row)

        tbl = ax.table(
            cellText=rows, colLabels=col_labels,
            cellLoc='center', loc='center',
            bbox=[0.0, 0.0, 1.0, 0.75]
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            is_header = (r == 0)
            is_mean   = (c == len(col_labels) - 1)
            bg = '#2a2a4e' if is_header else ('#1e3a2a' if is_mean else '#1a1a2e')
            cell.set_facecolor(bg)
            cell.set_text_props(color='white',
                fontweight='bold' if (is_mean and r > 0) else 'normal')
            cell.set_edgecolor('#334')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_table_per_cam(all_jitter, video_labels, out_dir):
    out_dir = Path(out_dir)
    metrics = [
        ('Fingertips only',        'fingertips_var'),
        ('All 21 joints',          'all_joints_var'),
        ('No thumb (excl. 1-4)',   'no_thumb_var'),
    ]
    col_labels = ['Model'] + list(video_labels) + ['Mean']

    def _style_table(tbl, n_cols):
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            is_header = (r == 0)
            is_mean   = (c == n_cols - 1)
            bg = '#2a2a4e' if is_header else ('#1e3a2a' if is_mean else '#1a1a2e')
            cell.set_facecolor(bg)
            cell.set_text_props(
                color='white',
                fontweight='bold' if (is_mean and r > 0) else 'normal')
            cell.set_edgecolor('#334')

    for cam_num in [1, 2, 3]:
        # One subplot per metric — no overlap guaranteed
        fig, axes = plt.subplots(len(metrics), 1,
                                  figsize=(14, 3.5 * len(metrics)))
        fig.patch.set_facecolor('#0d0d1a')
        fig.suptitle(
            f'Jitter Variance — Camera {cam_num} — relative to wrist\n'
            f'(m/frame)²,  HaMeR vs WiLoR',
            color='white', fontsize=13, y=1.02)

        for ax, (subtitle, metric_key) in zip(axes, metrics):
            ax.set_facecolor('#0d0d1a')
            ax.axis('off')
            # Title sits above the axes box, well clear of the table
            ax.set_title(subtitle, color='#aef', fontsize=11,
                         style='italic', loc='center', pad=8)

            rows = []
            for model_label, model in [('HaMeR', 'hamer'), ('WiLoR', 'wilor')]:
                row = [model_label]
                vals = []
                for vid in video_labels:
                    vid_key = vid.replace(' ', '_').lower()
                    k = f"{model}_cam{cam_num}_{vid_key}"
                    d = all_jitter.get(k, {})
                    sides = [d.get(s) for s in ['right', 'left'] if d.get(s)]
                    if sides:
                        v = np.mean([s[metric_key] for s in sides])
                        vals.append(v)
                        row.append(f'{v:.3e}')
                    else:
                        row.append('N/A')
                mean_v = np.mean(vals) if vals else None
                row.append(f'{mean_v:.3e}' if mean_v else 'N/A')
                rows.append(row)

            # bbox fills the lower 80% of the subplot leaving room for title
            tbl = ax.table(
                cellText=rows,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                bbox=[0.0, 0.05, 1.0, 0.80],
            )
            _style_table(tbl, len(col_labels))

        plt.tight_layout()
        save_path = out_dir / f'summary_table_cam{cam_num}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        plt.close()
        print(f"Saved: {save_path}")

def compute_cross_view_offset(preds_cam1, preds_cam2, preds_cam3, sorted_frames):
    """
    For each frame: compute mean XYZ offset of each non-cam2 view vs cam2.
    """
    offsets_1v2_r, offsets_1v2_l = [], []
    offsets_3v2_r, offsets_3v2_l = [], []

    for fnum in sorted_frames:
        for is_right, buf1, buf3 in [
                (True,  offsets_1v2_r, offsets_3v2_r),
                (False, offsets_1v2_l, offsets_3v2_l)]:

            j2 = preds_cam2.get(fnum, {}).get(is_right)
            if j2 is None: continue

            j1 = preds_cam1.get(fnum, {}).get(is_right)
            if j1 is not None:
                # Mean offset across all 21 joints
                diff = (j1 - j2).mean(axis=0)   # (3,) XYZ offset in meters
                buf1.append(diff)

            j3 = preds_cam3.get(fnum, {}).get(is_right)
            if j3 is not None:
                diff = (j3 - j2).mean(axis=0)
                buf3.append(diff)

    def summarize(offsets):
        if not offsets: return None
        arr = np.array(offsets)  # (F, 3)
        return {
            'mean_xyz':      arr.mean(axis=0).tolist(),   # systematic bias
            'std_xyz':       arr.std(axis=0).tolist(),    # random error
            'mean_dist':     float(np.linalg.norm(arr, axis=1).mean() * 100),  # cm
            'std_dist':      float(np.linalg.norm(arr, axis=1).std()  * 100),
            'p95_dist':      float(np.percentile(np.linalg.norm(arr, axis=1), 95) * 100),
            'n_frames':      len(offsets),
        }

    return {
        'cam1_vs_cam2': {'right': summarize(offsets_1v2_r),
                          'left':  summarize(offsets_1v2_l)},
        'cam3_vs_cam2': {'right': summarize(offsets_3v2_r),
                          'left':  summarize(offsets_3v2_l)},
    }

def plot_per_cam_depth(all_depth, video_labels, out_path):
    """
    Depth (Z) stats broken down per cam — one subplot per metric.
    """
    metrics = [
        ('mean_m',      'Mean depth (mm)'),
        ('std_m',       'Within-video depth std (mm)'),
        ('jitter_mean', 'Frame-to-frame depth jitter (mm)'),
    ]
    n_vids = len(video_labels)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0d0d1a')

    combos = [(m, c) for m in ['wilor','hamer'] for c in [1,2,3]]
    n = len(combos)
    x = np.arange(n_vids)
    width = 0.8 / n

    cam_shade = {1: 1.0, 2: 0.7, 3: 0.4}   # shade per cam within model color

    for ax, (metric_key, title) in zip(axes, metrics):
        _style(ax, title)
        for i, (model, cam_num) in enumerate(combos):
            base_color = COLORS_MODEL[model]
            # Darken/lighten by cam
            import matplotlib.colors as mcolors
            rgb = np.array(mcolors.to_rgb(base_color))
            shade = cam_shade[cam_num]
            color = tuple(np.clip(rgb * shade + (1-shade)*0.15, 0, 1))

            vals = []
            for vid in video_labels:
                vid_key = vid.replace(' ', '_').lower()
                k = f"{model}_cam{cam_num}_{vid_key}"
                d = all_depth.get(k, {})
                sides = [d.get(s) for s in ['right','left'] if d.get(s)]
                v = np.mean([s[metric_key] for s in sides]) if sides else 0
                vals.append(v * 1000)  # Convert to mm

            offset = (i - n/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=f"{model.upper()} c{cam_num}",
                          color=color, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(video_labels, color='#aaaaaa', fontsize=8)
        ax.set_ylabel('milimeters', color='#aaaaaa')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  ncol=2, loc='upper right')

    fig.suptitle('Depth Statistics — Per Camera View (WiLoR vs HaMeR)',
                 color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_cam_jitter(all_jitter, video_labels, out_path):
    """
    Per-joint jitter — one figure per model, one subplot per cam view.
    Rows = right/left hand. Cols = cam1/cam2/cam3.
    Averaged across videos.
    """
    for model in ['wilor', 'hamer']:
        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        fig.patch.set_facecolor('#0d0d1a')

        for cam_idx, cam_num in enumerate([1, 2, 3]):
            for side_idx, side in enumerate(['right', 'left']):
                ax = axes[side_idx, cam_idx]
                _style(ax, f'{model.upper()} cam{cam_num} — '
                           f'{side.capitalize()} Hand (mm/frame)')

                # Average per_joint_std across all videos for this cam
                arrays = []
                for vid in video_labels:
                    vid_key = vid.replace(' ', '_').lower()
                    k = f"{model}_cam{cam_num}_{vid_key}"
                    d = all_jitter.get(k, {})
                    if d.get(side):
                        arrays.append(d[side]['per_joint_std'])

                if not arrays:
                    ax.text(0.5, 0.5, 'No data', ha='center',
                            va='center', color='#aaa',
                            transform=ax.transAxes)
                    continue

                avg = np.mean(arrays, axis=0) * 1000.0  # (21,)
                std = np.std(arrays, axis=0) * 1000.0

                x = np.arange(21)
                color = COLORS_CAM[f'cam{cam_num}']
                ax.bar(x, avg, color=color, alpha=0.85, label='Mean across videos')
                ax.errorbar(x, avg, yerr=std, fmt='none',
                            ecolor='white', alpha=0.4, capsize=2)

                # Mark fingertips
                for ft in FINGERTIPS:
                    ax.axvline(ft, color='#ffffff', linewidth=0.8,
                               alpha=0.15, linestyle='--')

                ax.set_xticks(x)
                ax.set_xticklabels(JOINT_NAMES, rotation=45,
                                   ha='right', color='#aaaaaa', fontsize=6)
                ax.set_ylabel('Std (mm/frame)', color='#aaaaaa')

        fig.suptitle(f'{model.upper()} — Per-Joint Jitter Std per Camera View\n'
                     f'(dashed = fingertips, error bars = variance across videos)',
                     color='white', fontsize=13, y=1.01)
        plt.tight_layout()
        save_path = out_path.parent / f'per_joint_jitter_{model}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d0d1a')
        plt.close()
        print(f"Saved: {save_path}")


def plot_summary_jitter_per_cam(all_jitter, video_labels, out_path):
    """
    Summary bar chart: mean jitter variance per model+cam combo.
    One group per video, shows all 6 combos side by side.
    """
    combos = [(m, c) for m in ['wilor','hamer'] for c in [1,2,3]]
    cam_shade = {1: 1.0, 2: 0.7, 3: 0.4}
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor('#0d0d1a')

    x = np.arange(len(video_labels))
    n = len(combos)
    width = 0.8 / n

    for ax, (metric_key, title) in zip(axes, [
            ('fingertips_var', 'Fingertip jitter variance (m/frame)²'),
            ('all_joints_var', 'All-joints jitter variance (m/frame)²'),
            ('no_thumb_var',   'No-thumb jitter variance (m/frame)²')]):
        _style(ax, title)

        for i, (model, cam_num) in enumerate(combos):
            base_color = COLORS_MODEL[model]
            rgb   = np.array(mcolors.to_rgb(base_color))
            shade = cam_shade[cam_num]
            color = tuple(np.clip(rgb * shade + (1-shade)*0.15, 0, 1))

            vals = []
            for vid in video_labels:
                vid_key = vid.replace(' ', '_').lower()
                k = f"{model}_cam{cam_num}_{vid_key}"
                d = all_jitter.get(k, {})
                sides = [d.get(s) for s in ['right','left'] if d.get(s)]
                v = np.mean([s[metric_key] for s in sides]) if sides else 0
                vals.append(v)

            offset = (i - n/2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=f"{model.upper()} c{cam_num}",
                   color=color, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(video_labels, color='#aaaaaa')
        ax.set_ylabel('(m/frame)²', color='#aaaaaa')
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', ncol=2)

    fig.suptitle('Jitter Variance — All Camera Views (WiLoR vs HaMeR)',
                 color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"Saved: {out_path}")



def plot_per_cam_joint_jitter_separate(all_jitter, video_labels, out_dir):
    """
    3 separate figures (one per cam): WiLoR vs HaMeR per-joint jitter std.
    Each figure: 2 rows (right/left hand), bars = WiLoR vs HaMeR side by side.
    Averaged across videos.
    """
    out_dir = Path(out_dir)
    x = np.arange(21)
    width = 0.38

    for cam_num in [1, 2, 3]:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.patch.set_facecolor('#0d0d1a')

        for side_idx, side in enumerate(['right', 'left']):
            ax = axes[side_idx]
            _style(ax, f'Cam{cam_num} — {side.capitalize()} Hand — '
                       f'Per-Joint Jitter Std (mm/frame)')

            for mi, model in enumerate(['wilor', 'hamer']):
                arrays = []
                for vid in video_labels:
                    vid_key = vid.replace(' ', '_').lower()
                    k = f'{model}_cam{cam_num}_{vid_key}'
                    d = all_jitter.get(k, {})
                    if d.get(side):
                        arrays.append(d[side]['per_joint_std'])
                if not arrays:
                    continue
                avg = np.mean(arrays, axis=0) * 1000.0
                std = np.std(arrays,  axis=0) * 1000.0
                color = COLORS_MODEL[model]
                ax.bar(x + mi*width - width/2, avg, width,
                       label=f'{model.upper()}', color=color, alpha=0.85)
                ax.errorbar(x + mi*width - width/2, avg, yerr=std,
                            fmt='none', ecolor='white', alpha=0.35, capsize=2)

            for ft in FINGERTIPS:
                ax.axvline(ft, color='#ffffff', linewidth=0.8,
                           alpha=0.15, linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(JOINT_NAMES, rotation=45,
                               ha='right', color='#aaaaaa', fontsize=7)
            ax.set_ylabel('Std (mm/frame)', color='#aaaaaa')
            ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')

        fig.suptitle(f'Per-Joint Jitter Std — Camera {cam_num} — WiLoR vs HaMeR',
                     color='white', fontsize=13, y=1.01)
        plt.tight_layout()
        save_path = out_dir / f'joint_jitter_cam{cam_num}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        plt.close()
        print(f'Saved: {save_path}')


def plot_jitter_summary_per_cam_separate(all_jitter, video_labels, out_dir):
    """
    3 separate summary figures (one per cam).
    Each: 2 subplots (fingertips/all joints), WiLoR vs HaMeR bars per video.
    """
    out_dir = Path(out_dir)
    x = np.arange(len(video_labels))
    width = 0.35

    for cam_num in [1, 2, 3]:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.patch.set_facecolor('#0d0d1a')

        for ax, (metric_key, title) in zip(axes, [
                ('fingertips_var', 'Fingertip jitter variance (m/frame)²'),
                ('all_joints_var', 'All-joints jitter variance (m/frame)²'),
                ('no_thumb_var',   'No-thumb jitter variance (m/frame)²')]):
            _style(ax, title)

            for mi, model in enumerate(['wilor', 'hamer']):
                vals = []
                for vid in video_labels:
                    vid_key = vid.replace(' ', '_').lower()
                    k = f'{model}_cam{cam_num}_{vid_key}'
                    d = all_jitter.get(k, {})
                    sides = [d.get(s) for s in ['right','left'] if d.get(s)]
                    v = np.mean([s[metric_key] for s in sides]) if sides else 0
                    vals.append(v)
                ax.bar(x + mi*width, vals, width,
                       label=model.upper(), color=COLORS_MODEL[model], alpha=0.85)

            ax.set_xticks(x + width/2)
            ax.set_xticklabels(video_labels, color='#aaaaaa')
            ax.set_ylabel('(m/frame)²', color='#aaaaaa')

            # Mean line per model
            for mi, model in enumerate(['wilor', 'hamer']):
                vid_vals = []
                for vid in video_labels:
                    vid_key = vid.replace(' ', '_').lower()
                    k = f'{model}_cam{cam_num}_{vid_key}'
                    d = all_jitter.get(k, {})
                    sides = [d.get(s) for s in ['right','left'] if d.get(s)]
                    if sides:
                        vid_vals.append(np.mean([s[metric_key] for s in sides]))
                if vid_vals:
                    mean_v = np.mean(vid_vals)
                    ax.axhline(mean_v, color=COLORS_MODEL[model], linewidth=1.5,
                               linestyle='--', alpha=0.7,
                               label=f'{model.upper()} mean={mean_v:.2e}')
            ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

        fig.suptitle(f'Jitter Variance Summary — Camera {cam_num} — WiLoR vs HaMeR\n'
                     f'(dashed = mean across videos)',
                     color='white', fontsize=13)
        plt.tight_layout()
        save_path = out_dir / f'jitter_summary_cam{cam_num}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        plt.close()
        print(f'Saved: {save_path}')

def run(config):
    out = Path(config['out_folder'])
    out.mkdir(parents=True, exist_ok=True)

    with open(config['calib'], 'rb') as f:
        calib = pickle.load(f)

    K1=calib['cam1']['K'].astype(np.float64); R1=calib['cam1']['R'].astype(np.float64); T1=calib['cam1']['T'].flatten().astype(np.float64)
    K2=calib['cam2']['K'].astype(np.float64)
    K3=calib['cam3']['K'].astype(np.float64); R3=calib['cam3']['R'].astype(np.float64); T3=calib['cam3']['T'].flatten().astype(np.float64)

    limit       = config.get('limit', -1)
    video_labels = list(config['videos'].keys())

    all_jitter  = {}
    all_depth   = {}
    all_spread  = {}
    all_offsets = {}

    for vid_label, vid_cfg in config['videos'].items():
        img_w   = vid_cfg.get('img_w', 2704)
        img_h   = vid_cfg.get('img_h', 1520)
        f_syn   = 5000 / 256 * max(img_w, img_h)
        vid_key = vid_label.replace(' ', '_').lower()

        print(f"\n{'='*55}\nProcessing: {vid_label}\n{'='*55}")
        loaded = {}

        for model, loader_fn, cam_cfgs in [
            ('wilor', load_wilor, [
                (1,K1,R1,T1,f_syn),(2,K2,None,None,f_syn),(3,K3,R3,T3,f_syn)]),
            ('hamer', load_hamer, [
                (1,K1,R1,T1,None),(2,K2,None,None,None),(3,K3,R3,T3,None)])
        ]:
            for cam_args in cam_cfgs:
                cam_num = cam_args[0]
                path_key = f'{model}_cam{cam_num}'
                if not vid_cfg.get(path_key): continue
                print(f"  Loading {model.upper()} cam{cam_num}…")

                if model == 'wilor':
                    _, K, R, T, fs = cam_args
                    preds, frames = load_wilor(vid_cfg[path_key], K, fs,
                                               R=R, T=T, limit=limit)
                else:
                    _, K, R, T, _ = cam_args
                    preds, frames = load_hamer(vid_cfg[path_key], K,
                                               R=R, T=T, limit=limit)

                key = f"{model}_cam{cam_num}_{vid_key}"
                all_jitter[key] = compute_jitter_variance(preds, frames)
                all_depth[key]  = compute_depth_stats(preds, frames)
                loaded[f"{model}_cam{cam_num}"] = (preds, frames)
                print(f"    {len(preds)} frames")

        # Cross-view spread + offset
        for model in ['wilor', 'hamer']:
            c1 = loaded.get(f'{model}_cam1')
            c2 = loaded.get(f'{model}_cam2')
            c3 = loaded.get(f'{model}_cam3')
            if c1 and c2 and c3:
                print(f"  Computing {model.upper()} cross-view analysis…")
                common = sorted(set(c1[0]) & set(c2[0]) & set(c3[0]))
                all_spread[f"{model}_{vid_key}"] = compute_cross_view_spread(
                    c1[0], c2[0], c3[0], common)
                all_offsets[f"{model}_{vid_key}"] = compute_cross_view_offset(
                    c1[0], c2[0], c3[0], common)
                print(f"    {len(common)} common frames across all 3 cams")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    # Summary tables
    plot_summary_table(all_jitter, video_labels, out / 'summary_table.png')
    plot_summary_table_per_cam(all_jitter, video_labels, out)

    # Jitter bar charts
    plot_summary_jitter_per_cam(all_jitter, video_labels, out / 'jitter_per_cam_summary.png')
    plot_per_joint_jitter(all_jitter, video_labels, out / 'per_joint_jitter.png')
    plot_per_cam_jitter(all_jitter, video_labels, out / 'per_joint_jitter_per_cam.png')
    plot_per_cam_joint_jitter_separate(all_jitter, video_labels, out)
    plot_jitter_summary_per_cam_separate(all_jitter, video_labels, out)

    # Depth
    plot_per_cam_depth(all_depth, video_labels, out / 'depth_per_cam.png')
    plot_depth_comparison(all_depth, video_labels, out / 'depth_comparison.png')

    # Cross-view
    plot_cross_view_spread(all_spread, video_labels, out / 'cross_view_spread.png')

    # ── JSON ───────────────────────────────────────────────────────────────
    def ser(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.generic):  return o.item()
        if isinstance(o, dict):        return {k: ser(v) for k,v in o.items()}
        if isinstance(o, list):        return [ser(v) for v in o]
        return o

    with open(out / 'analysis_data.json', 'w') as f:
        json.dump(ser({'jitter': all_jitter, 'depth': all_depth,
                       'cross_view': all_spread,
                       'offsets': all_offsets}), f, indent=2)

    print(f"\n Done. Outputs: {out}")
    return all_jitter, all_depth, all_spread, all_offsets

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

# ── Example ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    config = {
        'calib':      '../calibration/calibration.pkl',
        'out_folder': '../analysis_extended',
        'limit':      -1,
        'videos': {
            'Video 1': {
                'wilor_cam1': '../data/1/test_p_n_w/predictions',
                'wilor_cam2': '../data/2/test_p_n_w/predictions',
                'wilor_cam3': '../data/3/test_p_n_w/predictions',
                'hamer_cam1': '../data/1/test_p_n_h/results.pkl',
                'hamer_cam2': '../data/2/test_p_n_h/results.pkl',
                'hamer_cam3': '../data/3/test_p_n_h/results.pkl',
                'img_w': 2704, 'img_h': 1520,
            },
            # 'Video 2': { ... },
            # 'Video 3': { ... },
        }
    }
    run(config)