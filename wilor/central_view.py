from pathlib import Path
import argparse, json, re
import numpy as np
import pickle

from wilor.models import load_wilor


def reconstruct_metric(verts_raw, cam_t_raw, kpts2d, is_right, f_real_x, f_real_y, cx, cy, f_syn):
    """
    Convert WiLoR output → metric OpenCV camera-space vertices.

    cam_t XY are derived from kpts2d wrist pixel position (exact unprojection).
    cam_t Z  is scaled by f_real/f_syn.
    Y is flipped to convert WiLoR Y-up → OpenCV Y-down.
    """
    # 1. Metric Z
    tz = cam_t_raw[2] * (f_real_x / f_syn)   # use fx for scale (symmetric enough)

    # 2. Unproject wrist pixel → metric XY  (exact, no normalization ambiguity)
    wrist_px = kpts2d[0]
    tx = (wrist_px[0] - cx) * tz / f_real_x
    ty = (wrist_px[1] - cy) * tz / f_real_y
    cam_t_metric = np.array([tx, ty, tz], dtype=np.float32)

    # 3. X-flip vertices for left hand (MANO is right-hand only)
    verts = verts_raw.copy().astype(np.float32)
    if not is_right:
        verts[:, 0] *= -1.0

    # 4. Compose: verts are local offsets around wrist in WiLoR cam space
    #    but their local XY also need the same flip for left hand
    #    verts[0] should be near origin (wrist), so add cam_t_metric as root
    verts_cam = verts + cam_t_metric   # still in WiLoR convention (Y-up)

    # 5. WiLoR → OpenCV: flip Y only (Z is positive-away in this data)
    verts_cam[:, 1] *= -1.0

    return verts_cam.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_folder', required=True)
    parser.add_argument('--calib_file',         required=True)
    parser.add_argument('--img_w', type=int,    required=True)
    parser.add_argument('--img_h', type=int,    required=True)
    parser.add_argument('--fps',   type=float,  default=240.0)
    parser.add_argument('--out_folder',         default='threejs_export')
    parser.add_argument('--limit', type=int,    default=-1)
    args = parser.parse_args()

    Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    # ── WiLoR config + MANO faces ───────────────────────────────────────────
    print("Loading WiLoR config …")
    model, cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path       ='./pretrained_models/model_config.yaml'
    )
    faces = model.mano.faces.astype(np.int32)
    f_syn = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(args.img_w, args.img_h)
    print(f"f_syn = {f_syn:.2f}")

    # calibration + frames loading
    with open(args.calib_file, 'rb') as f:
        calib = pickle.load(f)
    K      = calib['cam2']['K'].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    pred_dir = Path(args.predictions_folder)
    frames_dict = {}
    for p in pred_dir.glob('*_verts.npy'):
        m = re.search(r'frame_(\d+)_(\d+)_verts\.npy', p.name)
        if m:
            fnum = int(m.group(1))
            frames_dict.setdefault(fnum, []).append(m.group(2))

    sorted_frames = sorted(frames_dict.keys())
    if args.limit > 0:
        sorted_frames = sorted_frames[:args.limit]
    N = len(sorted_frames)
    V = 778
    print(f"Total frames: {N}")

    right_buf = np.full((N, V, 3), np.nan, dtype=np.float32)
    left_buf  = np.full((N, V, 3), np.nan, dtype=np.float32)
    frame_index_map = []

    for i, fnum in enumerate(sorted_frames):

        # When WiLoR detects 3+ hands, keep only highest-confidence per side
        best = {}  # {True: (score, hidx), False: (score, hidx)}
        for hidx in frames_dict[fnum]:
            score_path    = pred_dir / f'frame_{fnum:04d}_{hidx}_score.npy'
            is_right_path = pred_dir / f'frame_{fnum:04d}_{hidx}_is_right.npy'
            if not score_path.exists(): continue
            score    = float(np.load(score_path).ravel()[0])
            is_right = bool(np.load(is_right_path).ravel()[0])
            if is_right not in best or score > best[is_right][0]:
                best[is_right] = (score, hidx)

        has_right = False
        has_left  = False

        for is_right_raw, (score, hidx) in best.items():
            verts_path = pred_dir / f'frame_{fnum:04d}_{hidx}_verts.npy'
            cam_t_path = pred_dir / f'frame_{fnum:04d}_{hidx}_cam_t.npy'
            kpts_path  = pred_dir / f'frame_{fnum:04d}_{hidx}_kpts2d.npy'
            if not (verts_path.exists() and cam_t_path.exists() and kpts_path.exists()):
                continue

            verts_raw = np.load(verts_path).astype(np.float32)
            cam_t_raw = np.load(cam_t_path).astype(np.float32)
            kpts2d    = np.load(kpts_path).astype(np.float32)   # (778, 2) pixels

            # WiLoR is_right is mirrored relative to camera view — invert it
            is_right = not is_right_raw

            verts_cam = reconstruct_metric(
                verts_raw, cam_t_raw, kpts2d,
                is_right, fx, fy, cx, cy, f_syn
            )

            if is_right:
                # Strumming hand — position/rotation correct as-is
                right_buf[i] = verts_cam
                has_right = True
            else:
                # Fretting hand — mirror local shape around wrist (vertex 0)
                # Position (wrist) stays correct, only finger directions flip
                wrist_x = verts_cam[0, 0]
                verts_cam[:, 0] = 2 * wrist_x - verts_cam[:, 0]
                left_buf[i] = verts_cam
                has_left = True

        frame_index_map.append({'frame': fnum, 'right': has_right, 'left': has_left})

        if i % 500 == 0:
            print(f"  {i}/{N} frames …")

    # ── Write outputs ────────────────────────────────────────────────────────
    right_path = Path(args.out_folder) / 'hands_right.bin'
    left_path  = Path(args.out_folder) / 'hands_left.bin'
    mesh_path  = Path(args.out_folder) / 'hands_mesh.json'
    meta_path  = Path(args.out_folder) / 'hands_meta.json'

    right_buf.tofile(str(right_path))
    left_buf.tofile(str(left_path))
    print(f"Written {right_path}  ({right_buf.nbytes/1e6:.1f} MB)")
    print(f"Written {left_path}   ({left_buf.nbytes/1e6:.1f} MB)")

    with open(mesh_path, 'w') as f:
        json.dump({'faces': faces.tolist()}, f)

    meta = {
        'fps':        args.fps,
        'frameCount': N,
        'vertCount':  V,
        'faceCount':  len(faces),
        'imgWidth':   args.img_w,
        'imgHeight':  args.img_h,
        'camera': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
        'frameIndex': frame_index_map,
        'convention': 'OpenCV: X-right Y-down Z-away, metric meters'
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Written {mesh_path}")
    print(f"Written {meta_path}")
    print("\nDone!")

if __name__ == '__main__':
    main()
