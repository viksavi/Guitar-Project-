from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pickle
import time
from torchvision.transforms import functional as TF
from PIL import Image

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

torch.backends.cudnn.benchmark = True

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def apply_augmentations(img_tensor, num_augments=5):
    img_np = img_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    aug_images = []
    for _ in range(num_augments):
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.9, 1.1)
        aug_pil = TF.affine(img_pil, angle=angle, translate=(0, 0), scale=scale, shear=0)
        aug = TF.to_tensor(aug_pil)
        aug = TF.normalize(aug, mean=list(DEFAULT_MEAN), std=list(DEFAULT_STD))
        aug_images.append(aug)
    return torch.stack(aug_images)

def project_full_img(points_3d, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = np.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    pts = points_3d + cam_trans
    pts = pts / pts[..., 2:3]
    proj_2d = (K @ pts.T).T
    return proj_2d[:, :2]

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo with 2D keypoints & confidence')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--img_folder', type=str, default='images')
    parser.add_argument('--out_folder', type=str, default='out_demo')
    parser.add_argument('--side_view', action='store_true', default=False)
    parser.add_argument('--full_frame', action='store_true', default=True)
    parser.add_argument('--save_mesh', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet','regnety'])
    parser.add_argument('--file_type', nargs='+', default=['*.jpg','*.png'])
    parser.add_argument(
        '--joint_confidence',
        action='store_true',
        default=False,
        help='If set, compute per‐joint confidence via augmentations and include in results'
    )
    args = parser.parse_args()

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    detector.model.to(device)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    os.makedirs(args.out_folder, exist_ok=True)
    img_paths = [p for ext in args.file_type for p in Path(args.img_folder).glob(ext)]
    results = []

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        det_out = detector(img_cv2)
        img_rgb = img_cv2[:, :, ::-1]

        inst = det_out['instances']
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        bboxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        scores = inst.scores[keep].cpu().numpy()

        vit_outs = cpm.predict_pose(img_rgb, [np.concatenate([bboxes, scores[:, None]], axis=1)])
        hand_boxes, is_right = [], []
        vitpose_conf_per_hand = []

        for v in vit_outs:
            for hand_idx, hand_kpts in enumerate((v['keypoints'][-42:-21], v['keypoints'][-21:])):
                valid = hand_kpts[:, 2] > 0.5
                if valid.sum() > 3:
                    pts = hand_kpts[valid, :2]
                    x0, y0 = pts.min(axis=0)
                    x1, y1 = pts.max(axis=0)
                    hand_boxes.append([x0, y0, x1, y1])
                    is_right.append(hand_idx)
                    vitpose_conf_per_hand.append(hand_kpts[:, 2].tolist())
        if not hand_boxes:
            continue

        boxes = np.stack(hand_boxes)
        right = np.array(is_right, dtype=np.int64)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts, all_cam_t, all_right, all_joints_3d = [], [], [], []
        for batch in loader:
            batch = recursive_to(batch, device)
            for n in range(batch['img'].shape[0]):
                orig = batch['img'][n].unsqueeze(0)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        out = model({'img': orig})

                pred_j3d = out['pred_keypoints_3d'][0].detach().cpu().numpy()
                pm_params = {k: v[0].detach().cpu().numpy() for k, v in out['pred_mano_params'].items()}
                verts = out['pred_vertices'][0].detach().cpu().numpy()
                pred_cam_crop = out['pred_cam'][0].detach().cpu().numpy().copy()

                box_c = batch['box_center'][n].float().unsqueeze(0).to(device)
                box_s = batch['box_size'][n].float().unsqueeze(0).to(device)
                img_s = batch['img_size'][n].float().unsqueeze(0).to(device)
                f_len = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_s.max().item()

                m = (2 * batch['right'][n].item() - 1)
                pred_cam_crop[1] *= m

                cam_tensor = torch.from_numpy(pred_cam_crop)[None, :].to(device)
                pred_cam_full = cam_crop_to_full(cam_tensor, box_c, box_s, img_s, f_len).detach().cpu().numpy().squeeze()

                r = batch['right'][n].item()
                verts[:, 0] *= (2 * r - 1)
                pred_j3d[:, 0] *= (2 * r - 1)

                all_verts.append(verts)
                all_cam_t.append(pred_cam_full)
                all_right.append(r)
                all_joints_3d.append(pred_j3d)
                # ——— PROJECT ALL 3D JOINTS INTO 2D ———
                joints2d_all = project_full_img(
                    pred_j3d,       
                    pred_cam_full,
                    f_len,
                    img_s.squeeze().cpu().numpy()
                )

                # ——— OPTIONAL: augmentation‐based confidence ———
                conf_per_joint = None
                if args.joint_confidence:
                    aug_imgs = apply_augmentations(orig[0], num_augments=20).to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        with torch.no_grad():
                            out_aug = model({'img': aug_imgs})
                    # out_aug['pred_keypoints_3d']: (20, J, 3)
                    pred_joints_aug = out_aug['pred_keypoints_3d']
                    std_per_joint   = pred_joints_aug.std(dim=0)              # (J,3)
                    conf_per_joint  = 1.0 / (std_per_joint.norm(dim=1) + 1e-6) # (J,)
                    conf_per_joint  = conf_per_joint.cpu().numpy().tolist()

                results.append({
                    'image_name': os.path.basename(img_path),
                    'pred_joints_3d': pred_j3d,
                    'pred_cam_crop': pred_cam_crop.tolist(),
                    'pred_cam_full': pred_cam_full.tolist(),
                    'pred_mano_params': pm_params,
                    'joints2d_all': joints2d_all.tolist(),
                    'img_size':             img_s.squeeze().cpu().numpy().tolist(),
                    "is_right": r,
                    'confidence': conf_per_joint,
                    'vitpose_score': vitpose_conf_per_hand[n] if n < len(vitpose_conf_per_hand) else None
                })


        if args.full_frame and all_verts:
            f_len_full = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_cv2.shape[:2])
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=(img_cv2.shape[1], img_cv2.shape[0]),
                is_right=all_right, mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), focal_length=f_len_full
            )
            inp_full = img_cv2[:, :, ::-1].astype(np.float32) / 255.0
            inp_full = np.concatenate([inp_full, np.ones_like(inp_full[:, :, :1])], axis=2)
            overlay = inp_full[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            img_res = (img_cv2.shape[1], img_cv2.shape[0])  # width, height
            fingertip_indices = [4, 8, 12, 16, 20]
            for joints_3d, cam_t in zip(all_joints_3d, all_cam_t):
                fingertips_3d = joints_3d[fingertip_indices]
                joints_2d = project_full_img(fingertips_3d, cam_t, f_len_full, img_res)
                for x, y in joints_2d:
                    cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)

            fn, _ = os.path.splitext(os.path.basename(img_path))
            cv2.imwrite(os.path.join(args.out_folder, f"{fn}_full.png"), (255 * overlay[:, :, ::-1]).astype(np.uint8))

    with open(os.path.join(args.out_folder, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()