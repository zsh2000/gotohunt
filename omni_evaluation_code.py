# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import glob
import json
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mapanything.utils.image import load_images
from mapanything.utils.geometry import quaternion_to_rotation_matrix

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Model info registry: norm_type and resolution per model
# ──────────────────────────────────────────────────────────────────────────────

MODEL_INFO = {
    # Core models
    "mapanything":          {"norm_type": "dinov2",   "resolution": 518},
    "mapanything_ablations":{"norm_type": "dinov2",   "resolution": 518},
    "modular_dust3r":       {"norm_type": "dust3r",   "resolution": 512},
    # External models
    "pi3":      {"norm_type": "identity", "resolution": 518},
    "pi3x":     {"norm_type": "identity", "resolution": 518},
    "vggt":     {"norm_type": "identity", "resolution": 518},
    "dust3r":   {"norm_type": "dust3r",   "resolution": 512},
    "mast3r":   {"norm_type": "dust3r",   "resolution": 512},
    "must3r":   {"norm_type": "dust3r",   "resolution": 512},
    "pow3r":    {"norm_type": "dust3r",   "resolution": 512},
    "pow3r_ba": {"norm_type": "dust3r",   "resolution": 512},
    "da3":      {"norm_type": "dinov2",   "resolution": 504},
    "moge":     {"norm_type": "identity", "resolution": 518},
    "anycalib": {"norm_type": "identity", "resolution": 518},
}


# ──────────────────────────────────────────────────────────────────────────────
# 3D Reconstruction Metrics (from mv_recon/utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Mean distance from predicted points to nearest ground truth point."""
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)
    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals[idx] * rec_normals, axis=-1))
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Mean distance from ground truth points to nearest predicted point."""
    rec_points_kd_tree = KDTree(rec_points)
    distances, idx = rec_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals * rec_normals[idx], axis=-1))
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    """Percentage of GT points within distance threshold of predicted points."""
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    return np.mean((distances < dist_th).astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Pose Metrics (from relpose/evo_utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def c2w_to_tumpose(c2w):
    """Convert a cam-to-world 4x4 matrix to TUM format (x y z qw qx qy qz)."""
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    return np.concatenate([xyz, [qw, qx, qy, qz]])


def poses_to_tum_format(poses):
    """Convert list of 4x4 cam-to-world matrices to TUM trajectory format."""
    timestamps = np.arange(len(poses)).astype(float)
    tum_poses = np.stack([c2w_to_tumpose(p) for p in poses], axis=0)
    return tum_poses, timestamps


def eval_pose_metrics(pred_poses, gt_poses):
    """Evaluate ATE, RPE translation, and RPE rotation between predicted and GT poses."""
    try:
        import evo.main_ape as main_ape
        import evo.main_rpe as main_rpe
        from evo.core import sync
        from evo.core.metrics import PoseRelation, Unit
        from evo.core.trajectory import PoseTrajectory3D
    except ImportError:
        print("WARNING: 'evo' package not installed. Skipping pose metrics.")
        print("  Install with: pip install evo")
        return {"ate": float("nan"), "rpe_trans": float("nan"), "rpe_rot": float("nan")}

    pred_tum, pred_ts = poses_to_tum_format(pred_poses)
    gt_tum, gt_ts = poses_to_tum_format(gt_poses)

    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_tum[:, :3],
        orientations_quat_wxyz=pred_tum[:, 3:],
        timestamps=gt_ts,
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_tum[:, :3],
        orientations_quat_wxyz=gt_tum[:, 3:],
        timestamps=gt_ts,
    )

    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    ate_result = main_ape.ape(
        gt_traj, pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    ate = ate_result.stats["rmse"]

    rpe_rot_result = main_rpe.rpe(
        gt_traj, pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_rot = rpe_rot_result.stats["rmse"]

    rpe_trans_result = main_rpe.rpe(
        gt_traj, pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_trans = rpe_trans_result.stats["rmse"]

    return {
        "ate": ate,
        "rpe_trans": rpe_trans,
        "rpe_rot": rpe_rot,
        "ate_per_frame": ate_result.np_arrays["error_array"].tolist(),
        "rpe_trans_per_frame": rpe_trans_result.np_arrays["error_array"].tolist(),
        "rpe_rot_per_frame": rpe_rot_result.np_arrays["error_array"].tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Relative Pose Metrics (from mapanything/utils/metrics.py – no evo dependency)
# ──────────────────────────────────────────────────────────────────────────────

def relative_pose_error(pred_poses_4x4, gt_poses_4x4, max_auc_threshold=30):
    """Compute pairwise relative rotation/translation errors and AUC."""
    from mapanything.utils.metrics import (
        se3_to_relative_pose_error,
        calculate_auc_np,
    )

    if isinstance(pred_poses_4x4, np.ndarray):
        pred_poses_4x4 = torch.from_numpy(pred_poses_4x4).float()
    if isinstance(gt_poses_4x4, np.ndarray):
        gt_poses_4x4 = torch.from_numpy(gt_poses_4x4).float()

    N = pred_poses_4x4.shape[0]
    rot_err, trans_err = se3_to_relative_pose_error(pred_poses_4x4, gt_poses_4x4, N)

    rot_err_np = rot_err.cpu().numpy()
    trans_err_np = trans_err.cpu().numpy()

    auc, _ = calculate_auc_np(rot_err_np, trans_err_np, max_threshold=max_auc_threshold)

    return {"rot_errors": rot_err_np, "trans_errors": trans_err_np, "auc": auc}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-View Depth Estimation Metrics (adapted from StreamVGGT eval/video_depth)
# ──────────────────────────────────────────────────────────────────────────────

def depth2disparity(depth, return_mask=False):
    """Convert depth to disparity (1/depth), handling zeros."""
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negative_mask = depth > 0
    disparity[non_negative_mask] = 1.0 / depth[non_negative_mask]
    if return_mask:
        return disparity, non_negative_mask
    return disparity


def absolute_value_scaling2(
    predicted_depth,
    ground_truth_depth,
    s_init=1.0,
    t_init=0.0,
    lr=1e-4,
    max_iters=1000,
    tol=1e-6,
):
    """GPU-based L1 scale+shift alignment using Adam optimizer."""
    s = torch.tensor(
        [s_init], requires_grad=True,
        device=predicted_depth.device, dtype=predicted_depth.dtype,
    )
    t = torch.tensor(
        [t_init], requires_grad=True,
        device=predicted_depth.device, dtype=predicted_depth.dtype,
    )
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None
    for _ in range(max_iters):
        optimizer.zero_grad()
        predicted_aligned = s * predicted_depth + t
        loss = torch.sum(torch.abs(predicted_aligned - ground_truth_depth))
        loss.backward()
        optimizer.step()
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss.item()
    return s.detach().item(), t.detach().item()


def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
):
    """Evaluate predicted depth against ground truth using standard metrics.

    Supports multiple alignment modes: median scaling (default),
    scale&shift (align_with_lad2), scale-only (align_with_scale), or
    metric (no alignment).

    Returns a dict with: Abs Rel, Sq Rel, RMSE, Log RMSE,
    delta < 1.25, delta < 1.25^2, delta < 1.25^3, valid_pixels.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)

    # Flatten batch dimension if 3D
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)

    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter valid pixels (exclude zero/negative predictions from masked models)
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth) & (predicted_depth_original > 0)
    else:
        mask = (ground_truth_depth_original > 0) & (predicted_depth_original > 0)
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    num_valid_pixels = torch.sum(mask).item()
    if num_valid_pixels == 0:
        return {
            "Abs Rel": 0, "Sq Rel": 0, "RMSE": 0, "Log RMSE": 0,
            "delta < 1.25": 0, "delta < 1.25^2": 0, "delta < 1.25^3": 0,
            "valid_pixels": 0,
        }

    # Alignment
    if metric_scale:
        pass  # no alignment
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(
            predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Median scaling (matches Pi3/MonST3R official evaluation protocol)
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth = predicted_depth * scale_factor

    # Post-clip
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    # Compute metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()

    return {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "delta < 1.25": threshold_1,
        "delta < 1.25^2": threshold_2,
        "delta < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }


def extract_depth_from_pts3d(pts3d, pose):
    """Extract per-pixel depth from a 3D point map by transforming to camera frame.

    Args:
        pts3d: (H, W, 3) world-coordinate point map (numpy).
        pose: (4, 4) cam-to-world matrix (numpy).

    Returns:
        depth: (H, W) depth map (z-component in camera frame).
    """
    H, W = pts3d.shape[:2]
    w2c = np.linalg.inv(pose)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_flat = pts3d.reshape(-1, 3)
    pts_cam = (R @ pts_flat.T).T + t
    depth = pts_cam[:, 2].reshape(H, W)
    return depth


# ──────────────────────────────────────────────────────────────────────────────
# Dataset readers  (adapted from mv_recon/data.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_nrgbd_scene(scene_root, kf_every=1, max_frames=None):
    """Load images, depth maps, poses, and intrinsics for one Neural RGBD scene."""
    fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    num_files = len(os.listdir(os.path.join(scene_root, "images")))
    all_idxs = list(range(num_files))
    idxs = all_idxs[::kf_every]
    if max_frames is not None:
        idxs = idxs[:max_frames]

    with open(os.path.join(scene_root, "poses.txt")) as f:
        lines = f.readlines()
    all_poses = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if "nan" in lines[i]:
            all_poses.append(np.eye(4, dtype=np.float32))
        else:
            pose_floats = [
                [float(x) for x in line.split()] for line in lines[i : i + lines_per_matrix]
            ]
            all_poses.append(np.array(pose_floats, dtype=np.float32))

    image_paths, depth_maps, camera_poses = [], [], []
    for idx in idxs:
        impath = os.path.join(scene_root, "images", f"img{idx}.png")
        depthpath = os.path.join(scene_root, "depth", f"depth{idx}.png")

        depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
        depthmap[depthmap > 10] = 0
        depthmap[depthmap < 1e-3] = 0

        pose = all_poses[idx].copy()
        pose[:, 1:3] *= -1.0  # GL to CV

        image_paths.append(impath)
        depth_maps.append(depthmap)
        camera_poses.append(pose)

    return image_paths, depth_maps, camera_poses, intrinsics


def load_7scenes_scene(scene_root, kf_every=1, max_frames=None):
    """Load images, depth maps, poses, and intrinsics for one 7-Scenes sequence."""
    fx, fy, cx, cy = 525, 525, 320, 240
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    num_files = len([n for n in os.listdir(scene_root) if "color" in n])
    all_idxs = list(range(num_files))
    idxs = all_idxs[::kf_every]
    if max_frames is not None:
        idxs = idxs[:max_frames]

    image_paths, depth_maps, camera_poses = [], [], []
    for idx in idxs:
        impath = os.path.join(scene_root, f"frame-{idx:06d}.color.png")
        depthpath = os.path.join(scene_root, f"frame-{idx:06d}.depth.proj.png")
        posepath = os.path.join(scene_root, f"frame-{idx:06d}.pose.txt")

        depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap[depthmap == 65535] = 0
        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
        depthmap[depthmap > 10] = 0
        depthmap[depthmap < 1e-3] = 0

        camera_pose = np.loadtxt(posepath).astype(np.float32)

        image_paths.append(impath)
        depth_maps.append(depthmap)
        camera_poses.append(camera_pose)

    return image_paths, depth_maps, camera_poses, intrinsics


def _load_dtu_cam(path):
    """Parse MVSNet-style `_cam.txt`: returns (intrinsics 3x3, extrinsic 4x4 world-to-cam)."""
    with open(path, "r") as f:
        words = f.read().split()
    extr = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            extr[i, j] = float(words[4 * i + j + 1])
    intr = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            intr[i, j] = float(words[3 * i + j + 18])
    return intr, extr


def load_dtu_scene(scene_root, kf_every=1, max_frames=None):
    """Load images, depth maps, poses, and intrinsics for one DTU MVSNet-style scene.

    Expected layout:
        scene_root/
            images/        *.jpg
            depths/        {stem}.npy
            binary_masks/  {stem}.png
            cams/          {stem}_cam.txt   (MVSNet format)

    Units stay in native DTU millimetres (matching mv_recon reference: predictions
    are scaled ×1000 and `icp_threshold=100`). Depth is masked by the eroded
    binary foreground mask.
    """
    image_dir = os.path.join(scene_root, "images")
    depth_dir = os.path.join(scene_root, "depths")
    mask_dir = os.path.join(scene_root, "binary_masks")
    cam_dir = os.path.join(scene_root, "cams")

    img_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))
    img_files = img_files[::kf_every]
    if max_frames is not None:
        img_files = img_files[:max_frames]

    erosion_kernel = np.ones((10, 10), np.uint8)

    image_paths, depth_maps, camera_poses = [], [], []
    intrinsics_first = None
    for name in img_files:
        stem = os.path.splitext(name)[0]
        impath = os.path.join(image_dir, name)
        depthpath = os.path.join(depth_dir, stem + ".npy")
        maskpath = os.path.join(mask_dir, stem + ".png")
        campath = os.path.join(cam_dir, stem + "_cam.txt")

        depthmap = np.load(depthpath).astype(np.float32)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if mask.shape[:2] != depthmap.shape[:2]:
            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        mask = cv2.erode(mask, erosion_kernel, iterations=1)
        depthmap = depthmap * mask

        intr, extr_w2c = _load_dtu_cam(campath)
        pose = np.linalg.inv(extr_w2c).astype(np.float32)  # cam-to-world, mm

        if intrinsics_first is None:
            intrinsics_first = intr

        image_paths.append(impath)
        depth_maps.append(depthmap)
        camera_poses.append(pose)

    return image_paths, depth_maps, camera_poses, intrinsics_first


def load_tum_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one TUM-RGBD sequence."""
    fx, fy, cx, cy = 525, 525, 319.5, 239.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    for rgb_dir_name in ["rgb_90", "rgb_sync_for_pose_5_90", "rgb"]:
        rgb_dir = os.path.join(scene_root, rgb_dir_name)
        if os.path.isdir(rgb_dir):
            break
    else:
        raise FileNotFoundError(f"No rgb directory found in {scene_root}")

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    gt_poses = None
    for gt_name in ["groundtruth_90.txt", "groundtruth_sync_5_90.txt", "groundtruth.txt"]:
        gt_path = os.path.join(scene_root, gt_name)
        if os.path.exists(gt_path):
            try:
                from evo.tools import file_interface
                traj = file_interface.read_tum_trajectory_file(gt_path)
                poses_se3 = traj.poses_se3
                gt_poses = [np.array(p, dtype=np.float32) for p in poses_se3]
                gt_poses = gt_poses[::kf_every]
                if max_frames is not None:
                    gt_poses = gt_poses[:max_frames]
            except ImportError:
                print("WARNING: 'evo' package not installed, cannot load TUM trajectories.")
            break

    return image_paths, gt_poses, intrinsics


def load_long_tum_s1_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one long_tum_s1 sequence.

    Same as TUM but uses rgb_1000/ and groundtruth_1000.txt.
    """
    fx, fy, cx, cy = 525, 525, 319.5, 239.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgb_dir = os.path.join(scene_root, "rgb_1000")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"No rgb_1000 directory found in {scene_root}")

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    gt_poses = None
    gt_path = os.path.join(scene_root, "groundtruth_1000.txt")
    if os.path.exists(gt_path):
        try:
            from evo.tools import file_interface
            traj = file_interface.read_tum_trajectory_file(gt_path)
            poses_se3 = traj.poses_se3
            gt_poses = [np.array(p, dtype=np.float32) for p in poses_se3]
            gt_poses = gt_poses[::kf_every]
            if max_frames is not None:
                gt_poses = gt_poses[:max_frames]
        except ImportError:
            print("WARNING: 'evo' package not installed, cannot load TUM trajectories.")

    return image_paths, gt_poses, intrinsics


def load_long_tum_s1_150_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one long_tum_s1_150 sequence.

    Same as TUM but uses rgb_150/ and groundtruth_150.txt.
    """
    fx, fy, cx, cy = 525, 525, 319.5, 239.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgb_dir = os.path.join(scene_root, "rgb_150")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"No rgb_150 directory found in {scene_root}")

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    gt_poses = None
    gt_path = os.path.join(scene_root, "groundtruth_150.txt")
    if os.path.exists(gt_path):
        try:
            from evo.tools import file_interface
            traj = file_interface.read_tum_trajectory_file(gt_path)
            poses_se3 = traj.poses_se3
            gt_poses = [np.array(p, dtype=np.float32) for p in poses_se3]
            gt_poses = gt_poses[::kf_every]
            if max_frames is not None:
                gt_poses = gt_poses[:max_frames]
        except ImportError:
            print("WARNING: 'evo' package not installed, cannot load TUM trajectories.")

    return image_paths, gt_poses, intrinsics


def load_tum_full_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one tum_full sequence.

    Same as TUM but uses rgb/ and groundtruth.txt.
    """
    fx, fy, cx, cy = 525, 525, 319.5, 239.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgb_dir = os.path.join(scene_root, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"No rgb directory found in {scene_root}")

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    gt_poses = None
    gt_path = os.path.join(scene_root, "groundtruth.txt")
    if os.path.exists(gt_path):
        try:
            from evo.tools import file_interface
            traj = file_interface.read_tum_trajectory_file(gt_path)
            poses_se3 = traj.poses_se3
            gt_poses = [np.array(p, dtype=np.float32) for p in poses_se3]
            gt_poses = gt_poses[::kf_every]
            if max_frames is not None:
                gt_poses = gt_poses[:max_frames]
        except ImportError:
            print("WARNING: 'evo' package not installed, cannot load TUM trajectories.")

    return image_paths, gt_poses, intrinsics


def load_tum_mast3r_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one TUM-RGBD sequence (raw format).

    Reads images from rgb/ and ground truth from groundtruth.txt.
    Each image is matched to the closest ground-truth pose by timestamp.
    Image timestamps are extracted from filenames (e.g. 1305031102.175304.png).
    """
    fx, fy, cx, cy = 525, 525, 319.5, 239.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rgb_dir = os.path.join(scene_root, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"No rgb directory found in {scene_root}")

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    # Extract timestamps from image filenames
    def extract_timestamp(path):
        filename = os.path.basename(path)
        match = re.match(r'(\d+(?:\.\d+)?)', filename)
        if match:
            return float(match.group(1))
        raise ValueError(f"Cannot extract timestamp from filename: {filename}")

    image_timestamps = [extract_timestamp(p) for p in image_paths]

    # Parse groundtruth.txt and match by closest timestamp
    gt_poses = None
    gt_path = os.path.join(scene_root, "groundtruth.txt")
    if os.path.exists(gt_path):
        gt_timestamps = []
        gt_poses_all = []
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                rot = Rotation.from_quat([qx, qy, qz, qw])
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rot.as_matrix().astype(np.float32)
                pose[:3, 3] = [tx, ty, tz]

                gt_timestamps.append(ts)
                gt_poses_all.append(pose)

        gt_timestamps = np.array(gt_timestamps)

        # For each image, find the ground-truth pose with the closest timestamp
        gt_poses = []
        for img_ts in image_timestamps:
            idx = np.argmin(np.abs(gt_timestamps - img_ts))
            gt_poses.append(gt_poses_all[idx])

    return image_paths, gt_poses, intrinsics


def load_scannet_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory for one ScanNet sequence."""
    intrinsics = None

    color_dir = os.path.join(scene_root, "color_90")
    if not os.path.isdir(color_dir):
        color_dir = os.path.join(scene_root, "color")

    image_paths = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    image_paths = image_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    gt_poses = None
    for gt_name in ["pose_90.txt", "pose.txt"]:
        gt_path = os.path.join(scene_root, gt_name)
        if os.path.exists(gt_path):
            traj_w_c = np.loadtxt(gt_path)
            assert traj_w_c.shape[1] in (12, 16)
            if not np.all(np.isfinite(traj_w_c)):
                n_bad = int((~np.isfinite(traj_w_c).all(axis=1)).sum())
                raise ValueError(f"Pose file has {n_bad} frame(s) with NaN/Inf (failed tracking). Skipping scene.")
            gt_poses = []
            for r in traj_w_c:
                pose = np.array([
                    [r[0], r[1], r[2], r[3]],
                    [r[4], r[5], r[6], r[7]],
                    [r[8], r[9], r[10], r[11]],
                    [0, 0, 0, 1],
                ], dtype=np.float32)
                gt_poses.append(pose)
            gt_poses = gt_poses[::kf_every]
            if max_frames is not None:
                gt_poses = gt_poses[:max_frames]
            break

    return image_paths, gt_poses, intrinsics


def load_scannet_full_scene(scene_root, kf_every=1, max_frames=None):
    """Load images and GT trajectory from ordered_color + pose.txt (full ScanNet sequence)."""
    intrinsics = None

    color_dir = os.path.join(scene_root, "ordered_color")
    image_paths = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(color_dir, "*.png")))

    gt_poses = None
    gt_path = os.path.join(scene_root, "pose.txt")
    if os.path.exists(gt_path):
        traj_w_c = np.loadtxt(gt_path)
        assert traj_w_c.shape[1] in (12, 16)
        if not np.all(np.isfinite(traj_w_c)):
            n_bad = int((~np.isfinite(traj_w_c).all(axis=1)).sum())
            raise ValueError(f"Pose file has {n_bad} frame(s) with NaN/Inf (failed tracking). Skipping scene.")
        all_poses = []
        for r in traj_w_c:
            pose = np.array([
                [r[0], r[1], r[2], r[3]],
                [r[4], r[5], r[6], r[7]],
                [r[8], r[9], r[10], r[11]],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            all_poses.append(pose)

        n = min(len(image_paths), len(all_poses))
        image_paths, all_poses = image_paths[:n], all_poses[:n]

        image_paths = image_paths[::kf_every]
        gt_poses = all_poses[::kf_every]
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
            gt_poses = gt_poses[:max_frames]
    else:
        image_paths = image_paths[::kf_every]
        if max_frames is not None:
            image_paths = image_paths[:max_frames]

    return image_paths, gt_poses, intrinsics


# ──────────────────────────────────────────────────────────────────────────────
# Video depth dataset readers (Sintel, Bonn, KITTI)
# ──────────────────────────────────────────────────────────────────────────────

SINTEL_SEQ_LIST = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5", "market_6",
    "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3",
]


# SINTEL_SEQ_LIST = [
#     "alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5", "ambush_6", "ambush_7",
#     "bamboo_1", "bamboo_2", "bandage_1", "bandage_2",
#     "cave_2", "cave_4", "market_2", "market_5", "market_6",
#     "mountain_1", "shaman_2", "shaman_3",
#     "sleeping_1", "sleeping_2", "temple_2", "temple_3",
# ]

BONN_SEQ_LIST = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

# bonn_full: all 26 sequences using full-length rgb/ and depth/ dirs (TUM timestamp-based)
BONN_FULL_SEQ_LIST = [
    "balloon", "balloon2", "balloon_tracking", "balloon_tracking2",
    "crowd", "crowd2", "crowd3",
    "kidnapping_box", "kidnapping_box2",
    "moving_nonobstructing_box", "moving_nonobstructing_box2",
    "moving_obstructing_box", "moving_obstructing_box2",
    "person_tracking", "person_tracking2",
    "placing_nonobstructing_box", "placing_nonobstructing_box2", "placing_nonobstructing_box3",
    "placing_obstructing_box",
    "removing_nonobstructing_box", "removing_nonobstructing_box2",
    "removing_obstructing_box",
    "static", "static_close_far",
    "synchronous", "synchronous2",
]


def _sintel_depth_read(filename):
    """Read Sintel .dpt depth file (custom binary float32 format)."""
    TAG_FLOAT = 202021.25
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, (
            f"depth_read: Wrong tag (expected {TAG_FLOAT}, got {check}). Big-endian?"
        )
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size < 100000000, (
            f"depth_read: Wrong size (w={width}, h={height})"
        )
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def _bonn_depth_read(filename):
    """Read Bonn 16-bit PNG depth (divide by 5000)."""
    depth_png = np.asarray(Image.open(filename))
    assert np.max(depth_png) > 255, "Expected 16-bit depth"
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = 0.0
    return depth


def _kitti_depth_read(filename):
    """Read KITTI 16-bit PNG depth (divide by 256)."""
    depth_png = np.array(Image.open(filename), dtype=int)
    assert np.max(depth_png) > 255, "Expected 16-bit depth"
    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = 0.0
    return depth


def load_sintel_scene(data_root, seq_name, kf_every=1, max_frames=None):
    """Load images and GT depth for one Sintel sequence.

    Args:
        data_root: Sintel root directory (contains training/final/ and training/depth/).
        seq_name: Sequence name, e.g. "alley_2".

    Returns:
        (image_paths, gt_depth_maps, None, None)
    """
    img_dir = os.path.join(data_root, "training", "final", seq_name)
    depth_dir = os.path.join(data_root, "training", "depth", seq_name)

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.dpt")))

    # Align counts
    n = min(len(image_paths), len(depth_paths))
    image_paths = image_paths[:n]
    depth_paths = depth_paths[:n]

    # Subsample
    image_paths = image_paths[::kf_every]
    depth_paths = depth_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        depth_paths = depth_paths[:max_frames]

    gt_depth_maps = [_sintel_depth_read(p) for p in depth_paths]

    return image_paths, gt_depth_maps, None, None


def load_bonn_scene(data_root, seq_name, kf_every=1, max_frames=None):
    """Load images and GT depth for one Bonn sequence.

    Args:
        data_root: Bonn root directory (contains rgbd_bonn_{seq}/).
        seq_name: Sequence name, e.g. "balloon2".

    Returns:
        (image_paths, gt_depth_maps, None, None)
    """
    seq_dir = os.path.join(data_root, f"rgbd_bonn_{seq_name}")
    img_dir = os.path.join(seq_dir, "rgb_110")
    depth_dir = os.path.join(seq_dir, "depth_110")

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    n = min(len(image_paths), len(depth_paths))
    image_paths = image_paths[:n]
    depth_paths = depth_paths[:n]

    image_paths = image_paths[::kf_every]
    depth_paths = depth_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        depth_paths = depth_paths[:max_frames]

    gt_depth_maps = [_bonn_depth_read(p) for p in depth_paths]

    return image_paths, gt_depth_maps, None, None


def _tum_associate(rgb_txt, depth_txt, max_dt=0.02):
    """Associate RGB and depth frames by closest timestamps (TUM format).

    Args:
        rgb_txt: Path to rgb.txt (timestamp + path per line).
        depth_txt: Path to depth.txt.
        max_dt: Maximum timestamp difference for a valid match (seconds).

    Returns:
        List of (rgb_path, depth_path) tuples.
    """
    def _parse(txt_path):
        entries = []
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                entries.append((float(parts[0]), parts[1]))
        return entries

    rgb_entries = _parse(rgb_txt)
    depth_entries = _parse(depth_txt)

    pairs = []
    di = 0
    for rt, rp in rgb_entries:
        # Find closest depth timestamp
        best_dt = float("inf")
        best_dp = None
        while di < len(depth_entries):
            dt_val = abs(depth_entries[di][0] - rt)
            if dt_val < best_dt:
                best_dt = dt_val
                best_dp = depth_entries[di][1]
                di += 1
            else:
                # We've passed the closest match; back up one
                di = max(0, di - 1)
                break
        if best_dp is not None and best_dt <= max_dt:
            pairs.append((rp, best_dp))

    return pairs


def load_bonn_full_scene(data_root, seq_name, kf_every=1, max_frames=None):
    """Load images and GT depth for one Bonn full-length sequence.

    Uses TUM timestamp association (rgb.txt + depth.txt) to match frames.

    Args:
        data_root: Bonn dataset root (contains rgbd_bonn_{seq}/).
        seq_name: Sequence name, e.g. "balloon2".

    Returns:
        (image_paths, gt_depth_maps, None, None)
    """
    seq_dir = os.path.join(data_root, f"rgbd_bonn_{seq_name}")
    rgb_txt = os.path.join(seq_dir, "rgb.txt")
    depth_txt = os.path.join(seq_dir, "depth.txt")

    pairs = _tum_associate(rgb_txt, depth_txt)

    image_paths = [os.path.join(seq_dir, rp) for rp, _ in pairs]
    depth_paths = [os.path.join(seq_dir, dp) for _, dp in pairs]

    image_paths = image_paths[::kf_every]
    depth_paths = depth_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        depth_paths = depth_paths[:max_frames]

    gt_depth_maps = [_bonn_depth_read(p) for p in depth_paths]

    return image_paths, gt_depth_maps, None, None


def load_kitti_scene(data_root, seq_name, kf_every=1, max_frames=None):
    """Load images and GT depth for one KITTI depth selection sequence.

    Args:
        data_root: KITTI val_selection_cropped root (contains image_gathered/ and
                   groundtruth_depth_gathered/).
        seq_name: Sequence directory name.

    Returns:
        (image_paths, gt_depth_maps, None, None)
    """
    img_dir = os.path.join(data_root, "image_gathered", seq_name)
    depth_dir = os.path.join(data_root, "groundtruth_depth_gathered", seq_name)

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    n = min(len(image_paths), len(depth_paths))
    image_paths = image_paths[:n]
    depth_paths = depth_paths[:n]

    image_paths = image_paths[::kf_every]
    depth_paths = depth_paths[::kf_every]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        depth_paths = depth_paths[:max_frames]

    gt_depth_maps = [_kitti_depth_read(p) for p in depth_paths]

    return image_paths, gt_depth_maps, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Depth back-projection helper
# ──────────────────────────────────────────────────────────────────────────────

def backproject_depth(depth, intrinsics, camera_pose=None):
    """Back-project a depth map to 3D points.

    If camera_pose (c2w) is provided, returns world-coordinate points.
    If camera_pose is None, returns camera-coordinate points.
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=-1)

    if camera_pose is None:
        return pts_cam

    pts_flat = pts_cam.reshape(-1, 3)
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    pts_world_flat = (R @ pts_flat.T).T + t
    return pts_world_flat.reshape(H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Umeyama alignment
# ──────────────────────────────────────────────────────────────────────────────

def umeyama_alignment(src, dst, with_scale=True):
    """Compute optimal scale, rotation, translation via Umeyama SVD."""
    assert src.shape == dst.shape
    N, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    Sigma = dst_c.T @ src_c / N
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / N
        s = (D * S.diagonal()).sum() / var_src
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    return s, R, t


# ──────────────────────────────────────────────────────────────────────────────
# 3D Reconstruction evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _scale_shift_align(pred_pts_list, gt_pts_list, masks_list):
    """Replicate the Regr3D_t_ScaleShiftInv alignment from the reference.

    Both pred and GT point map lists should already be in camera-1's frame.
    Steps (following criterion.py MRO: ShiftInv first, then ScaleInv):
      1. Shift: subtract per-cloud median z-depth
      2. Scale: scale pred so its median norm matches GT's median norm
    """
    # ── Step 1: Shift-invariant (subtract median z) ──────────────────────
    # Collect all valid z values
    gt_zs, pred_zs = [], []
    for gt, pr, m in zip(gt_pts_list, pred_pts_list, masks_list):
        gt_zs.append(gt[m][..., 2])
        pred_zs.append(pr[m][..., 2])
    gt_z_all = np.concatenate(gt_zs)
    pred_z_all = np.concatenate(pred_zs)

    gt_shift_z = np.nanmedian(gt_z_all)
    pred_shift_z = np.nanmedian(pred_z_all)

    gt_pts_shifted = [g.copy() for g in gt_pts_list]
    pred_pts_shifted = [p.copy() for p in pred_pts_list]
    for g in gt_pts_shifted:
        g[..., 2] -= gt_shift_z
    for p in pred_pts_shifted:
        p[..., 2] -= pred_shift_z

    # ── Step 2: Scale-invariant (match median norm, gt_scale=True) ───────
    # Compute median norm relative to median center
    gt_valid = np.concatenate([g[m] for g, m in zip(gt_pts_shifted, masks_list)])
    pred_valid = np.concatenate([p[m] for p, m in zip(pred_pts_shifted, masks_list)])

    gt_center = np.nanmedian(gt_valid, axis=0, keepdims=True)
    pred_center = np.nanmedian(pred_valid, axis=0, keepdims=True)

    gt_scale = np.nanmedian(np.linalg.norm(gt_valid - gt_center, axis=-1))
    pred_scale = np.nanmedian(np.linalg.norm(pred_valid - pred_center, axis=-1))
    pred_scale = np.clip(pred_scale, 1e-3, 1e3)

    # gt_scale=True in the reference: pred *= gt_scale / pred_scale
    scale_factor = gt_scale / pred_scale
    for p in pred_pts_shifted:
        p *= scale_factor

    print(f"    [3D eval] Scale-shift alignment: "
          f"gt_shift_z={gt_shift_z:.4f}, pred_shift_z={pred_shift_z:.4f}, "
          f"gt_scale={gt_scale:.4f}, pred_scale={pred_scale:.4f}, "
          f"scale_factor={scale_factor:.4f}")

    return pred_pts_shifted, gt_pts_shifted


def evaluate_3d_reconstruction(
    pred_pts3d_list,
    gt_depth_maps,
    gt_intrinsics,
    gt_poses,
    pred_poses=None,
    icp_threshold=0.1,
    use_umeyama=False,
    crop_center=224,
):
    """Evaluate 3D reconstruction quality (accuracy, completion, normal consistency).

    Both pred and GT are transformed to their respective camera-1 frames:
      - GT: backproject depth → GT world (using accurate GT poses),
            then inv(gt_poses[0]) → GT camera-1 frame.  Clean and consistent.
      - Pred: model world coords, then inv(pred_poses[0]) → model camera-1 frame.
            Only ONE pose is used, so pose error is a global offset → ICP handles it.
    Scale-shift alignment + ICP bridges the gap between the two camera-1 frames.

    Why camera-1 is better than world coordinates:
      World-coord approach uses pred_poses[i] per view to place GT, so per-view pose
      errors corrupt and scatter the GT cloud.  Camera-1 keeps GT pristine (GT poses
      only) and confines model pose error to a single global transform.

    Steps:
      1. GT → GT-camera-1;  Pred → model-camera-1
      2. Scale-shift invariant alignment
      3. Center-crop (default 224x224)
      4. ICP refinement
      5. Metrics (accuracy, completion, normal consistency)
    """
    n_views = len(pred_pts3d_list)
    print(f"    [3D eval] Transforming to camera-1 frames ({n_views} views) ...")

    # GT: world → GT-camera-1 (uses accurate GT poses)
    gt_w2c1 = np.linalg.inv(gt_poses[0])
    R_gt = gt_w2c1[:3, :3]
    t_gt = gt_w2c1[:3, 3]

    # Pred: model-world → model-camera-1 (uses single pred pose → global offset only)
    if pred_poses is not None and len(pred_poses) > 0:
        pred_w2c1 = np.linalg.inv(pred_poses[0])
        R_pred = pred_w2c1[:3, :3]
        t_pred = pred_w2c1[:3, 3]
        transform_pred = True
        print(f"    [3D eval] Pred: inv(pred_poses[0]) → model camera-1")
    else:
        transform_pred = False
        print(f"    [3D eval] No pred_poses — assuming pred already in camera-1 frame")

    pred_maps, gt_maps, mask_maps = [], [], []

    for i in range(n_views):
        pred = pred_pts3d_list[i]
        pred_H, pred_W = pred.shape[:2]

        gt_depth = gt_depth_maps[i]
        gt_H, gt_W = gt_depth.shape[:2]
        if (gt_H, gt_W) != (pred_H, pred_W):
            scale_y = pred_H / gt_H
            scale_x = pred_W / gt_W
            gt_depth_resized = cv2.resize(gt_depth, (pred_W, pred_H), interpolation=cv2.INTER_NEAREST)
            adjusted_intrinsics = gt_intrinsics.copy()
            adjusted_intrinsics[0, 0] *= scale_x
            adjusted_intrinsics[1, 1] *= scale_y
            adjusted_intrinsics[0, 2] *= scale_x
            adjusted_intrinsics[1, 2] *= scale_y
        else:
            gt_depth_resized = gt_depth
            adjusted_intrinsics = gt_intrinsics

        # GT: depth → world (accurate GT pose) → GT-camera-1
        gt_pts_world = backproject_depth(gt_depth_resized, adjusted_intrinsics, gt_poses[i])
        H, W = gt_pts_world.shape[:2]
        gt_flat = gt_pts_world.reshape(-1, 3)
        gt_cam1 = (R_gt @ gt_flat.T).T + t_gt
        gt_cam1 = gt_cam1.reshape(H, W, 3)

        # Pred: model-world → model-camera-1
        if transform_pred:
            pred_flat = pred.reshape(-1, 3)
            pred_cam1 = (R_pred @ pred_flat.T).T + t_pred
            pred_cam1 = pred_cam1.reshape(pred_H, pred_W, 3)
        else:
            pred_cam1 = pred

        # GT depth validity mask
        mask = gt_depth_resized > 0

        pred_maps.append(pred_cam1)
        gt_maps.append(gt_cam1)
        mask_maps.append(mask)

    # ── Scale-shift invariant alignment (before crop, matching reference) ─
    print(f"    [3D eval] Applying scale-shift invariant alignment ...")
    pred_maps, gt_maps = _scale_shift_align(pred_maps, gt_maps, mask_maps)

    # ── Center crop + collect masked points ──────────────────────────────
    pts_all_list, pts_gt_all_list = [], []
    for i in range(n_views):
        pred = pred_maps[i]
        gt_pts = gt_maps[i]
        mask = mask_maps[i]

        if crop_center is not None:
            half = crop_center // 2
            H, W = pred.shape[:2]
            cy, cx = H // 2, W // 2
            t, b = cy - half, cy + half
            l, r = cx - half, cx + half
            pred = pred[t:b, l:r]
            gt_pts = gt_pts[t:b, l:r]
            mask = mask[t:b, l:r]

        pts_all_list.append(pred[mask])
        pts_gt_all_list.append(gt_pts[mask])

    pts_all = np.concatenate(pts_all_list, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all_list, axis=0)

    finite_mask = np.isfinite(pts_all).all(axis=1)
    pts_all = pts_all[finite_mask]
    pts_gt_all = pts_gt_all[finite_mask]
    print(f"    [3D eval] {len(pts_all):,} pred points, {len(pts_gt_all):,} GT points after filtering")

    if len(pts_all) == 0:
        return {"acc": float("nan"), "comp": float("nan"), "chamfer": float("nan"),
                "nc1": float("nan"), "nc2": float("nan"), "nc": float("nan")}

    if use_umeyama:
        print(f"    [3D eval] Umeyama alignment (scale + rotation + translation) ...")
        s, R, t = umeyama_alignment(pts_all, pts_gt_all, with_scale=True)
        pts_all = (s * (R @ pts_all.T)).T + t

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pts_all)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all)

    print(f"    [3D eval] Running ICP refinement ...")
    t0 = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_pred = pcd_pred.transform(reg_p2p.transformation)
    print(f"    [3D eval] ICP done in {time.time() - t0:.1f}s")

    print(f"    [3D eval] Estimating normals ...")
    t0 = time.time()
    pcd_pred.estimate_normals()
    pcd_gt.estimate_normals()
    print(f"    [3D eval] Normals done in {time.time() - t0:.1f}s")

    gt_normals = np.asarray(pcd_gt.normals)
    pred_normals = np.asarray(pcd_pred.normals)

    print(f"    [3D eval] Computing accuracy (KDTree query) ...")
    t0 = time.time()
    acc, acc_med, nc1, nc1_med = accuracy(
        np.asarray(pcd_gt.points), np.asarray(pcd_pred.points), gt_normals, pred_normals
    )
    print(f"    [3D eval] Accuracy done in {time.time() - t0:.1f}s")

    print(f"    [3D eval] Computing completion (KDTree query) ...")
    t0 = time.time()
    comp, comp_med, nc2, nc2_med = completion(
        np.asarray(pcd_gt.points), np.asarray(pcd_pred.points), gt_normals, pred_normals
    )
    print(f"    [3D eval] Completion done in {time.time() - t0:.1f}s")

    chamfer = (acc + comp) / 2.0
    nc = (nc1 + nc2) / 2.0

    return {
        "acc": acc, "acc_med": acc_med,
        "comp": comp, "comp_med": comp_med,
        "chamfer": chamfer,
        "nc1": nc1, "nc1_med": nc1_med,
        "nc2": nc2, "nc2_med": nc2_med,
        "nc": nc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Unified inference wrapper
# ──────────────────────────────────────────────────────────────────────────────

def _build_camera_poses_4x4(cam_trans, cam_quats):
    """Construct (B, 4, 4) cam-to-world matrices from translation and quaternion."""
    B = cam_trans.shape[0]
    poses = torch.eye(4, device=cam_trans.device, dtype=cam_trans.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    rot = quaternion_to_rotation_matrix(cam_quats)
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = cam_trans
    return poses


def standardize_predictions(predictions):
    """Ensure every prediction dict has 'camera_poses' and 'mask' keys.

    Works with output from any model wrapper (Pi3, VGGT, DUSt3R, etc.).
    """
    standardized = []
    for pred in predictions:
        std_pred = dict(pred)

        # Build camera_poses (4x4) from cam_trans + cam_quats if missing
        if "camera_poses" not in std_pred:
            if "cam_trans" in std_pred and "cam_quats" in std_pred:
                std_pred["camera_poses"] = _build_camera_poses_4x4(
                    std_pred["cam_trans"].float(), std_pred["cam_quats"].float()
                )

        # Build a mask if missing
        if "mask" not in std_pred:
            if "conf" in std_pred:
                conf = std_pred["conf"]
                if conf.dim() == 3:  # (B, H, W)
                    std_pred["mask"] = (conf > 0).unsqueeze(-1).float()
                else:  # (B, H, W, 1)
                    std_pred["mask"] = (conf > 0).float()
            elif "non_ambiguous_mask" in std_pred:
                mask = std_pred["non_ambiguous_mask"]
                if mask.dim() == 3:  # (B, H, W)
                    std_pred["mask"] = mask.unsqueeze(-1).float()
                else:
                    std_pred["mask"] = mask.float()
            else:
                # Fallback: all pixels valid
                pts = std_pred["pts3d"]
                std_pred["mask"] = torch.ones(*pts.shape[:-1], 1, device=pts.device)

        standardized.append(std_pred)
    return standardized


def load_covisibility_matrix(covisibility_root, scene_name, num_frames):
    """Load a pre-computed covisibility matrix for a scene.

    Searches for covisibility_matrix.npy under covisibility_root using the
    scene_name path (e.g. "chess/seq-01" or "fr1_desk").

    Args:
        covisibility_root: Root directory containing per-scene matrices.
        scene_name: Scene identifier (may include subdirectories).
        num_frames: Expected number of frames (for validation).

    Returns:
        np.ndarray of shape (num_frames, num_frames) or None.
    """
    if covisibility_root is None:
        return None

    matrix_path = os.path.join(covisibility_root, scene_name, "covisibility_matrix.npy")
    if not os.path.exists(matrix_path):
        # Fall back to similarity_matrix.npy (e.g. from MegaLoc)
        matrix_path = os.path.join(covisibility_root, scene_name, "similarity_matrix.npy")
    if not os.path.exists(matrix_path):
        print(f"  [COVISIBILITY] Matrix not found at {matrix_path}, skipping.")
        return None

    covis = np.load(matrix_path).astype(np.float32)
    if covis.ndim != 2 or covis.shape[0] != covis.shape[1]:
        print(f"  [COVISIBILITY] Invalid shape {covis.shape}, skipping.")
        return None

    N = covis.shape[0]
    if N == num_frames:
        print(f"  [COVISIBILITY] Loaded {N}x{N} matrix.")
        return covis

    if N > num_frames:
        # Matrix covers more frames than we selected; slice to first num_frames
        covis = covis[:num_frames, :num_frames]
        print(f"  [COVISIBILITY] Sliced {N}x{N} -> {num_frames}x{num_frames}.")
        return covis

    # Matrix is smaller than the number of frames — cannot use
    print(f"  [COVISIBILITY] Matrix size {N} < num_frames {num_frames}, skipping.")
    return None


def _load_layer_config(path):
    """Load per-layer config from a YAML file. Returns dict {layer_idx: {overrides}} or None."""
    if path is None:
        return None
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None or "layers" not in raw:
        return None
    # Normalize keys to int
    config = {}
    for k, v in raw["layers"].items():
        config[int(k)] = v
    print(f"[LayerConfig] Loaded per-layer config for {len(config)} layers from {path}")
    return config


def build_frame_selector(args, covisibility_matrix=None):
    """Build a frame_selector dict from CLI arguments, or None if not requested."""
    if args.frame_strategy is None:
        return None
    selector = {
        "strategy": args.frame_strategy,
        "top_k": args.frame_topk,
        "include_self": args.frame_include_self,
        "include_first": args.frame_include_first,
        "token_downsample": args.frame_token_downsample,
        "token_keep_rate": args.token_keep_rate,
        "token_selection_method": args.token_selection_method,
        "token_ds_layers": args.token_ds_layers,
        "token_ds_entropy_threshold": args.token_ds_entropy_threshold,
        "ds_keep_register": args.ds_keep_register,
        "use_covisibility": args.frame_covisibility and covisibility_matrix is not None,
        "covisibility_percentile": args.frame_covisibility_percentile,
        "batched_sdpa": args.batched_sdpa,
        "global_as_frame_layers": set(args.global_as_frame_layers) if args.global_as_frame_layers else None,
        "global_as_meanpool_layers": set(args.global_as_meanpool_layers) if args.global_as_meanpool_layers else None,
    }
    # 'diverse' always needs the covisibility matrix; topk/even need it only as pre-filter
    if selector["use_covisibility"] or args.frame_strategy in ("diverse", "diverse_self"):
        if covisibility_matrix is not None:
            selector["covisibility_matrix"] = covisibility_matrix
        elif args.frame_strategy in ("diverse", "diverse_self"):
            raise ValueError(
                f"--frame_strategy {args.frame_strategy} requires --covisibility_root "
                "to provide the covisibility matrix"
            )
    return selector


def run_inference(model, model_name, views, device, args, covisibility_matrix=None, scene_name=None):
    """Run inference through either MapAnything .infer() or generic forward().

    Returns a list of standardized prediction dicts (one per view) with at
    least the keys: pts3d, camera_poses, mask.
    """
    # MapAnything has a rich .infer() method — use it for best results
    if model_name in ("mapanything", "mapanything_ablations") and hasattr(model, "infer"):
        predictions = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient,
            minibatch_size=args.minibatch_size,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            apply_mask=args.apply_mask,
            mask_edges=args.mask_edges,
            apply_confidence_mask=args.apply_confidence_mask,
            confidence_percentile=args.confidence_percentile,
        )
        return predictions  # already has camera_poses and mask

    # For all external models: move views to device and call forward()
    for view in views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

    # Build frame_selector from CLI args (Pi3/Pi3X/VGGT, ignored by other models)
    frame_selector = build_frame_selector(args, covisibility_matrix=covisibility_matrix)

    with torch.no_grad():
        extra_kwargs = {}
        if frame_selector is not None:
            extra_kwargs["frame_selector"] = frame_selector
        elif args.frame_token_downsample > 1 or args.token_keep_rate < 1.0:
            # token_downsample / token_keep_rate can be used independently of frame_strategy
            extra_kwargs["token_downsample"] = args.frame_token_downsample
            extra_kwargs["token_keep_rate"] = args.token_keep_rate
            extra_kwargs["token_selection_method"] = args.token_selection_method
            extra_kwargs["ds_keep_register"] = args.ds_keep_register
        if args.backbone_minibatch_size > 0:
            extra_kwargs["backbone_minibatch_size"] = args.backbone_minibatch_size
        # Per-layer config from YAML file (overrides CLI args above)
        layer_config = _load_layer_config(getattr(args, 'layer_config', None))
        if layer_config is None:
            layer_config = {}

        # Merge --global_as_frame_layers / --global_as_meanpool_layers into layer_config
        # CLI args set the strategy; layer_config YAML can override further.
        if getattr(args, 'global_as_frame_layers', None):
            for idx in args.global_as_frame_layers:
                layer_config.setdefault(idx, {}).setdefault("strategy", "frame")
        if getattr(args, 'global_as_meanpool_layers', None):
            for idx in args.global_as_meanpool_layers:
                layer_config.setdefault(idx, {}).setdefault("strategy", "meanpool")

        if layer_config:
            extra_kwargs["layer_config"] = layer_config

        # Adaptive entropy-based routing thresholds (supported by VGGT and Pi3)
        if model_name in ("vggt", "pi3"):
            if getattr(args, "adaptive_ds_threshold", None) is not None:
                extra_kwargs["adaptive_ds_threshold"] = args.adaptive_ds_threshold
            if getattr(args, "adaptive_frame_threshold", None) is not None:
                extra_kwargs["adaptive_frame_threshold"] = args.adaptive_frame_threshold

        # Attention-temperature (dilution) probe (Pi3 only for now)
        if model_name == "pi3":
            if getattr(args, "tau", 1.0) != 1.0:
                extra_kwargs["tau"] = args.tau
                if getattr(args, "tau_layers", None) is not None:
                    extra_kwargs["tau_layers"] = args.tau_layers

        if getattr(args, "diagnose_attention", False):
            from diagnose_attention import AttentionDiagnostics
            diag = AttentionDiagnostics(
                sample_query_frames=5,
                sample_heads=4,
                sample_q_tokens=50,
                sample_layers=getattr(args, "diagnose_attention_layers", None),
            )
            diag.install()
            predictions = model(views, **extra_kwargs)
            diag.uninstall()
            diag.print_summary()
            # Save lightweight CSV summary (no raw attention weights)
            out_dir = getattr(args, "output_dir", None)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                suffix = f"_{scene_name.replace('/', '_')}" if scene_name else ""
                plot_path = os.path.join(out_dir, f"attention_diag{suffix}.png")
                diag.save_plots(plot_path)
                csv_path = os.path.join(out_dir, f"attention_summary{suffix}.csv")
                diag.save_summary_csv(csv_path)
            # Store summary rows on args for cross-scene aggregation
            args._attention_diag = diag
        else:
            predictions = model(views, **extra_kwargs)

    return standardize_predictions(predictions)


def _clear_model_caches(model):
    """Clear internal GPU tensor caches (e.g. VGGT RoPE position/frequency caches).

    These caches persist across forward passes and can cause OOM on subsequent scenes.
    """
    for module in model.modules():
        # RotaryPositionEmbedding2D.frequency_cache (nn.Module, found by .modules())
        if hasattr(module, "frequency_cache") and isinstance(module.frequency_cache, dict):
            module.frequency_cache.clear()
        # PositionGetter is a plain class stored as an attribute, not an nn.Module
        if hasattr(module, "position_getter") and hasattr(module.position_getter, "position_cache"):
            module.position_getter.position_cache.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Scene discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_scenes(data_root, dataset_type):
    """Return a list of scene root directories."""
    if dataset_type == "images":
        return [("scene", data_root)]

    if dataset_type == "nrgbd":
        scenes = sorted([
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ])
        return [(s, os.path.join(data_root, s)) for s in scenes]

    if dataset_type == "7scenes":
        result = []
        for scene in sorted(os.listdir(data_root)):
            scene_path = os.path.join(data_root, scene)
            if not os.path.isdir(scene_path):
                continue
            test_split_file = os.path.join(scene_path, "TestSplit.txt")
            if os.path.exists(test_split_file):
                with open(test_split_file) as f:
                    seq_ids = f.read().splitlines()
                for seq_id in seq_ids:
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_name = f"seq-{num_part.zfill(2)}"
                    seq_path = os.path.join(scene_path, seq_name)
                    if os.path.isdir(seq_path):
                        result.append((f"{scene}/{seq_name}", seq_path))
            else:
                for seq_dir in sorted(os.listdir(scene_path)):
                    if seq_dir.startswith("seq-"):
                        result.append((f"{scene}/{seq_dir}", os.path.join(scene_path, seq_dir)))
        return result

    if dataset_type in ("tum", "tum_full", "tum_mast3r", "long_tum_s1", "long_tum_s1_150", "scannet", "scannet_full", "dtu"):
        scenes = sorted([
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ])
        return [(s, os.path.join(data_root, s)) for s in scenes]

    # Video depth datasets: scene_root is the shared data_root for all sequences
    if dataset_type == "sintel":
        # Use the standard 14-sequence eval subset (MonST3R protocol, Pi3 Table 4)
        return [(s, data_root) for s in SINTEL_SEQ_LIST]

    if dataset_type == "bonn":
        return [(s, data_root) for s in BONN_SEQ_LIST]

    if dataset_type == "bonn_full":
        # Same 5 evaluation sequences as bonn, but using full-length rgb/depth
        return [(s, data_root) for s in BONN_SEQ_LIST]

    if dataset_type == "kitti":
        img_dir = os.path.join(data_root, "image_gathered")
        seqs = sorted([
            d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))
        ])
        return [(s, data_root) for s in seqs]

    raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate any supported model on various datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mapanything",
        help=(
            "Model name from the registry: "
            + ", ".join(sorted(MODEL_INFO.keys()))
            + ". Or a HuggingFace path like 'facebook/map-anything' "
            "(treated as MapAnything)."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the evaluation dataset (or image folder for --dataset images)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nrgbd",
        choices=["nrgbd", "7scenes", "tum", "tum_full", "tum_mast3r", "long_tum_s1", "long_tum_s1_150", "scannet", "scannet_full", "sintel", "bonn", "bonn_full", "kitti", "dtu", "images"],
        help="Dataset type",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Parent directory for results. A subdirectory '<model>/<dataset>' "
             "is created automatically (e.g. eval_results/pi3/tum/).",
    )
    parser.add_argument(
        "--kf_every",
        type=int,
        default=5,
        help="Keyframe sampling stride",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames per scene",
    )
    parser.add_argument(
        "--icp_threshold",
        type=float,
        default=0.1,
        help="ICP distance threshold (use 100 for DTU-scale datasets)",
    )
    parser.add_argument(
        "--use_umeyama",
        action="store_true",
        default=False,
        help="Apply Umeyama alignment (scale+rotation+translation) before ICP. "
             "Use when model predictions are in a different coordinate frame than GT.",
    )
    parser.add_argument(
        "--crop_center",
        type=int,
        default=224,
        help="Crop to this center size before computing metrics (default: 224, matching "
             "StreamVGGT/VGGT protocol). Set to 0 to disable.",
    )
    parser.add_argument(
        "--eval_3d",
        action="store_true",
        default=False,
        help="Evaluate 3D reconstruction metrics (requires GT depth + poses)",
    )
    parser.add_argument(
        "--eval_pose",
        action="store_true",
        default=False,
        help="Evaluate camera pose metrics (ATE, RPE) using the evo library",
    )
    parser.add_argument(
        "--eval_relpose",
        action="store_true",
        default=False,
        help="Evaluate pairwise relative pose errors and AUC (no evo dependency)",
    )
    parser.add_argument(
        "--eval_depth",
        action="store_true",
        default=False,
        help="Evaluate multi-view depth estimation metrics (requires GT depth maps + predicted poses)",
    )
    parser.add_argument(
        "--depth_align",
        type=str,
        default="scale",
        choices=["scale&shift", "scale", "metric"],
        help="Alignment mode for depth evaluation: "
             "'scale' (median scaling, matches Pi3/MonST3R protocol, default), "
             "'scale&shift' (L1 scale+shift via Adam), "
             "'metric' (no alignment, assumes metric-scale predictions)",
    )
    parser.add_argument(
        "--depth_max_depth",
        type=float,
        default=None,
        help="Maximum depth threshold for depth evaluation (meters). "
             "GT pixels beyond this are ignored. Default: dataset-dependent (10 for nrgbd/7scenes).",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="fixed_mapping",
        choices=["fixed_mapping", "longest_side", "square", "fixed_size", "fixed_width"],
        help="Image resize mode. 'fixed_mapping' uses predefined aspect ratio buckets (default). "
             "'longest_side' resizes longest side to --resize_size preserving aspect ratio. "
             "'fixed_width' sets width to --resize_size, scales height proportionally (matches "
             "recons_eval/Pi3 official with --resize_size 512). "
             "For sintel/kitti/bonn depth eval, use --resize_mode fixed_width --resize_size 512.",
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=None,
        help="Target size for 'longest_side' or 'square' resize mode. "
             "E.g., --resize_mode longest_side --resize_size 518.",
    )
    parser.add_argument(
        "--save_pointclouds",
        action="store_true",
        default=False,
        help="Save predicted and GT point clouds as PLY files",
    )
    parser.add_argument(
        "--save_ply_recons_eval",
        action="store_true",
        default=False,
        help="Save {scene}-pred.ply and {scene}-gt.ply in the flat output dir, "
             "using recons_eval/mv_recon/eval.py conventions: Umeyama-align pred to "
             "GT, filter by GT valid mask, include RGB colors. Requires GT depth + "
             "poses + intrinsics.",
    )

    # MapAnything-specific inference arguments (ignored for other models)
    parser.add_argument("--memory_efficient", action="store_true", default=True)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--apply_mask", action="store_true", default=True)
    parser.add_argument("--mask_edges", action="store_true", default=True)
    parser.add_argument("--apply_confidence_mask", action="store_true", default=False)
    parser.add_argument("--confidence_percentile", type=int, default=10)
    parser.add_argument("--scenes", nargs="+", default=None, help="Specific scene names to evaluate")

    # Frame selection arguments (for Pi3/Pi3X sparse cross-view attention)
    parser.add_argument(
        "--frame_strategy",
        type=str,
        default=None,
        choices=["topk", "topk_mean", "incvggt_max", "incvggt_mean", "even", "diverse", "diverse_self", "random", "closest"],
        help="Frame selection strategy for sparse cross-view attention. "
             "'topk' uses cosine-similarity max-pool of Q/K over patch pairs. "
             "'topk_mean' is the same but mean-pooled instead of max. "
             "'incvggt_max' uses raw scaled Q@K^T logits max-pooled over key/"
             "query tokens and heads (IncVGGT-style, adapted to batch mode). "
             "'incvggt_mean' is the same but mean-pooled instead of max. "
             "'diverse' uses FPS on covisibility to maximize viewpoint diversity. "
             "'closest' picks the K frames closest in index to the query frame "
             "(K/2 before + K/2 after, compensating from the other side). "
             "If not set, uses full attention (default Pi3 behavior).",
    )
    parser.add_argument(
        "--frame_topk",
        type=int,
        default=10,
        help="Number of frames each query frame attends to (default: 10)",
    )
    parser.add_argument(
        "--frame_include_self",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the query frame itself in the selected frames (default: True)",
    )
    parser.add_argument(
        "--frame_include_first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Always include frame 0 (the first frame) in the selected frames (default: False)",
    )
    parser.add_argument(
        "--frame_covisibility",
        action="store_true",
        default=False,
        help="Pre-filter candidate frames using covisibility before applying topk/even",
    )
    parser.add_argument(
        "--frame_covisibility_percentile",
        type=float,
        default=50.0,
        help="Keep the top-X%% of frames by shared 3D points (default: 50.0)",
    )
    parser.add_argument(
        "--frame_token_downsample",
        type=int,
        default=1,
        help="Spatial stride factor for downsampling K/V tokens in cross-frame attention. "
             "E.g., 2 means tokens_h and tokens_w are each strided by 2, yielding ~1/4 tokens. "
             "Works with or without --frame_strategy. Default: 1 (no downsampling).",
    )
    parser.add_argument(
        "--ds_keep_register",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the query frame itself in the selected frames (default: True)",
    )
    parser.add_argument(
        "--token_keep_rate",
        type=float,
        default=1.0,
        help="Fraction of patch tokens to keep per frame via diverse (FPS) token selection. "
             "In (0, 1]. E.g., 0.25 keeps 25%% of patch tokens per frame. "
             "Mutually exclusive with --frame_token_downsample > 1. Default: 1.0 (keep all).",
    )
    parser.add_argument(
        "--token_selection_method",
        type=str,
        choices=["diverse", "activation", "per_frame_diverse", "per_frame_activation", "per_token_activation"],
        default="diverse",
        help="Method for selecting which tokens to keep when token_keep_rate < 1.0. "
             "'diverse': FPS on key features averaged across frames (same tokens for all frames). "
             "'activation': Top-k by L2 norm averaged across frames (same tokens for all frames). "
             "'per_frame_diverse': FPS independently per frame (each frame keeps its own tokens). "
             "'per_frame_activation': Top-k by L2 norm independently per frame. "
             "'per_token_activation': Full Q@K^T then per-query-token top-k selection (most accurate, slowest). "
             "Default: 'diverse'.",
    )
    parser.add_argument(
        "--token_ds_layers",
        nargs="+",
        type=int,
        default=None,
        help="Which global (cross-view) attention layers should apply intra-frame token "
             "downsampling (0-indexed). Default: None = all global layers. "
             "E.g., --token_ds_layers 0 3 5 applies token_keep_rate only to those layers.",
    )
    parser.add_argument(
        "--token_ds_entropy_threshold",
        type=float,
        default=None,
        help="Entropy-based adaptive token downsampling. When set, each global layer's "
             "Ent/Max ratio is computed on-the-fly during inference. Layers with "
             "Ent/Max >= threshold apply token downsampling; others skip it. "
             "This is per-scene adaptive (each scene may select different layers). "
             "Overrides --token_ds_layers when both are set. E.g., --token_ds_entropy_threshold 0.93",
    )
    parser.add_argument(
        "--global_as_frame_layers",
        nargs="+",
        type=int,
        default=None,
        help="Convert these global attention layers (0-indexed) into frame attention. "
             "Skips the (B*S,P,C)->(B,S*P,C) rearrangement so attention runs independently "
             "per frame using the global block's weights. Reduces cost from O((NL)^2) to "
             "O(NL^2) for the affected layers. E.g., --global_as_frame_layers 0 1 2 3 "
             "converts the first 4 global layers. Default: None (no conversion).",
    )
    parser.add_argument(
        "--adaptive_ds_threshold",
        type=float,
        default=None,
        help="Adaptive entropy threshold for token downsampling. Layers are processed "
             "sequentially; as long as Ent/Max >= this threshold, token downsampling "
             "(token_keep_rate / token_downsample) is applied. Once a layer drops below, "
             "all subsequent layers use full attention. "
             "Overridden by --adaptive_frame_threshold for layers with even higher entropy. "
             "E.g., --adaptive_ds_threshold 0.90",
    )
    parser.add_argument(
        "--adaptive_frame_threshold",
        type=float,
        default=None,
        help="Adaptive entropy threshold for frame attention routing. Must be >= "
             "--adaptive_ds_threshold. Layers with Ent/Max >= this threshold are routed "
             "to frame-only attention (global_as_frame). Layers between this and "
             "--adaptive_ds_threshold get token downsampling. Layers below "
             "--adaptive_ds_threshold get full attention. Transitions are one-directional: "
             "once a layer exits a regime, it cannot re-enter. "
             "E.g., --adaptive_frame_threshold 0.95 --adaptive_ds_threshold 0.90",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Attention-softmax temperature for the dilution probe (Pi3 only). "
             "softmax((Q K^T) * scale / tau). tau<1 sharpens (less dilution), "
             "tau>1 flattens (control). tau=1.0 is a no-op and uses the fused "
             "SDPA fast path. Suggested sweep: 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0",
    )
    parser.add_argument(
        "--tau_layers",
        nargs="+",
        type=int,
        default=None,
        help="Cross-view layer indices (0-indexed in [0, dec_depth/2); for Pi3-large "
             "this is 0..17) to which --tau is applied. Default: None = all "
             "cross-view layers. E.g. --tau_layers 9 10 11 12 13 14 15 16 17 "
             "applies tau only to the late half.",
    )
    parser.add_argument(
        "--global_as_meanpool_layers",
        nargs="+",
        type=int,
        default=None,
        help="Replace these global attention layers (0-indexed) with mean-pooling of V. "
             "For early layers where attention is nearly uniform, output ≈ mean(V). "
             "Skips Q/K projection entirely. Cost: O(NL) instead of O((NL)^2). "
             "E.g., --global_as_meanpool_layers 0 1 2 3. Default: None.",
    )
    parser.add_argument(
        "--batched_sdpa",
        action="store_true",
        default=False,
        help="Use a single batched SDPA call instead of per-frame loop for sparse "
             "cross-view attention. Higher memory but eliminates bf16 rounding "
             "divergence. Only effective when --frame_strategy is set.",
    )
    parser.add_argument(
        "--backbone_minibatch_size",
        type=int,
        default=0,
        help="Process DINOv2 backbone in mini-batches of this size (0 = all at once). "
             "Reduces peak GPU memory at the cost of slightly more compute time.",
    )
    parser.add_argument(
        "--covisibility_root",
        type=str,
        default=None,
        help="Root directory for pre-computed covisibility matrices. "
             "Expected layout: {root}/{scene_name}/covisibility_matrix.npy",
    )
    parser.add_argument(
        "--layer_config",
        type=str,
        default=None,
        help="Path to a YAML config file with per-layer overrides for global attention. "
             "Supports per-layer: strategy (meanpool/frame/global), token_keep_rate, "
             "token_selection_method. See configs/layer_config_example.yaml.",
    )
    parser.add_argument(
        "--diagnose_attention",
        action="store_true",
        default=False,
        help="Enable attention diagnostics: print and visualize per-layer attention "
             "entropy, per-frame mass concentration, and inter-token coherence. "
             "Saves plots to {output_dir}/attention_diag_{scene}.png.",
    )
    parser.add_argument(
        "--diagnose_attention_layers",
        nargs="+",
        type=int,
        default=None,
        help="Which global attention layers to diagnose (0-indexed). "
             "Default: None = all layers. E.g., --diagnose_attention_layers 0 5 11",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        default=False,
        help="Skip evaluation metrics (pose/3D/depth) after inference. "
             "Useful with --diagnose_attention to only inspect attention patterns.",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if not (args.eval_3d or args.eval_pose or args.eval_relpose or args.eval_depth):
        print("No evaluation mode selected. Specify at least one of:")
        print("  --eval_3d      (3D reconstruction: accuracy, completion, normal consistency)")
        print("  --eval_pose    (trajectory: ATE, RPE via evo library)")
        print("  --eval_relpose (pairwise relative pose error & AUC)")
        print("  --eval_depth   (multi-view depth estimation: Abs Rel, RMSE, delta thresholds)")
        return

    # Handle crop_center=0 as disable
    if args.crop_center == 0:
        args.crop_center = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Resolve model name and info ───────────────────────────────────────
    model_name = args.model
    is_hf_path = "/" in model_name  # e.g. "facebook/map-anything"

    if is_hf_path:
        # Treat HuggingFace paths as MapAnything
        print(f"Loading MapAnything from HuggingFace: {model_name} ...")
        from mapanything.models import MapAnything
        model = MapAnything.from_pretrained(model_name).to(device)
        info = MODEL_INFO["mapanything"]
        model_name = "mapanything"
    else:
        if model_name not in MODEL_INFO:
            available = ", ".join(sorted(MODEL_INFO.keys()))
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )
        info = MODEL_INFO[model_name]
        print(f"Loading model '{model_name}' via model factory ...")
        from mapanything.models import init_model_from_config
        model = init_model_from_config(model_name, device=device)

    model.eval()
    norm_type = info["norm_type"]
    resolution = info["resolution"]
    print(f"  norm_type={norm_type}, resolution={resolution}")

    # Check capabilities for the requested evaluations
    has_poses = model_name not in ("moge",)
    if not has_poses and (args.eval_pose or args.eval_relpose):
        print(f"WARNING: Model '{model_name}' does not predict camera poses. "
              "Pose evaluation will be skipped.")

    # ── Build output directory: <output_dir>/<model>/<dataset> ────────────
    output_dir = os.path.join(args.output_dir, model_name, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # ── Save config: command line + all args ─────────────────────────────
    config_path = os.path.join(output_dir, "config.json")
    config = {
        "command": " ".join(sys.argv),
        "args": vars(args),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    # ── Auto-detect DTU and override ICP threshold (matching mv_recon) ─────
    if "DTU" in args.data_root or "dtu" in args.data_root.lower():
        if args.icp_threshold == 0.1:  # only override if user didn't set explicitly
            args.icp_threshold = 100
            print(f"Auto-detected DTU dataset: setting icp_threshold=100")

    # ── Discover scenes ─────────────────────────────────────────────────────
    scenes = discover_scenes(args.data_root, args.dataset)
    if args.scenes is not None:
        scenes = [(n, p) for n, p in scenes if n in args.scenes]
    print(f"Found {len(scenes)} scene(s) to evaluate.")

    all_results = {}
    skipped_scenes = {}  # scene_name -> reason string
    all_attn_summaries = {}  # scene_name -> list of summary row dicts

    for scene_name, scene_root in tqdm(scenes[:], desc="Evaluating scenes"):
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

        try:
            # ── Load dataset ────────────────────────────────────────────────
            gt_depth_maps, gt_poses, gt_intrinsics = None, None, None

            if args.dataset == "nrgbd":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_nrgbd_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "7scenes":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_7scenes_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "dtu":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_dtu_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "tum":
                image_paths, gt_poses, gt_intrinsics = load_tum_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "tum_full":
                image_paths, gt_poses, gt_intrinsics = load_tum_full_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "tum_mast3r":
                image_paths, gt_poses, gt_intrinsics = load_tum_mast3r_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "long_tum_s1":
                image_paths, gt_poses, gt_intrinsics = load_long_tum_s1_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "long_tum_s1_150":
                image_paths, gt_poses, gt_intrinsics = load_long_tum_s1_150_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "scannet":
                image_paths, gt_poses, gt_intrinsics = load_scannet_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "scannet_full":
                image_paths, gt_poses, gt_intrinsics = load_scannet_full_scene(
                    scene_root, kf_every=args.kf_every, max_frames=args.max_frames
                )
                gt_depth_maps = None
            elif args.dataset == "sintel":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_sintel_scene(
                    scene_root, scene_name, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "bonn":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_bonn_scene(
                    scene_root, scene_name, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "bonn_full":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_bonn_full_scene(
                    scene_root, scene_name, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "kitti":
                image_paths, gt_depth_maps, gt_poses, gt_intrinsics = load_kitti_scene(
                    scene_root, scene_name, kf_every=args.kf_every, max_frames=args.max_frames
                )
            elif args.dataset == "images":
                image_paths = sorted(
                    glob.glob(os.path.join(scene_root, "*.png"))
                    + glob.glob(os.path.join(scene_root, "*.jpg"))
                    + glob.glob(os.path.join(scene_root, "*.jpeg"))
                )
                image_paths = image_paths[:: args.kf_every]
                if args.max_frames is not None:
                    image_paths = image_paths[: args.max_frames]
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")

            if len(image_paths) == 0:
                print(f"  No images found, skipping.")
                continue

            print(f"  {len(image_paths)} images found")

            # ── Load images with model-appropriate normalization ───────────
            print(f"  Preprocessing images (norm={norm_type}, res={resolution}) ...")
            t0 = time.time()
            views = load_images(
                image_paths,
                norm_type=norm_type,
                resolution_set=resolution,
                resize_mode=getattr(args, 'resize_mode', 'fixed_mapping'),
                size=getattr(args, 'resize_size', None),
            )
            print(f"  Preprocessing done in {time.time() - t0:.1f}s")

            # ── Load covisibility matrix if requested ─────────────────────
            covis_matrix = None
            needs_covis = args.frame_covisibility or args.frame_strategy in ("diverse", "diverse_self")
            if needs_covis and args.covisibility_root is not None:
                covis_matrix = load_covisibility_matrix(
                    args.covisibility_root, scene_name, num_frames=len(views)
                )

            # ── Run inference ──────────────────────────────────────────────
            print(f"  Running {model_name} inference on {len(views)} views ...")
            t0 = time.time()
            predictions = run_inference(model, model_name, views, device, args,
                                        covisibility_matrix=covis_matrix,
                                        scene_name=scene_name)
            torch.cuda.synchronize() if device == "cuda" else None
            print(f"  Inference done in {time.time() - t0:.1f}s")

            scene_results = {"scene": scene_name, "num_views": len(image_paths)}

            if getattr(args, "skip_metrics", False):
                print("  --skip_metrics: skipping evaluation metrics.")
                all_results[scene_name] = scene_results
                del predictions, views
                _clear_model_caches(model)
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # ── Extract predicted outputs ──────────────────────────────────
            print(f"  Extracting predictions ...")
            pred_pts3d_list = []
            pred_pts3d_cam_list = []
            pred_poses_list = []
            pred_masks_list = []

            for pred in predictions:
                pred_pts3d_list.append(pred["pts3d"].cpu().numpy()[0])       # (H, W, 3)
                if "pts3d_cam" in pred:
                    pred_pts3d_cam_list.append(pred["pts3d_cam"].cpu().numpy()[0])  # (H, W, 3)
                if "camera_poses" in pred:
                    pred_poses_list.append(pred["camera_poses"].cpu().numpy()[0])  # (4, 4)
                pred_masks_list.append(pred["mask"].cpu().numpy()[0, ..., 0] > 0.5)  # (H, W)
            # Free GPU tensors immediately after extracting numpy arrays
            del pred, predictions, views
            _clear_model_caches(model)
            gc.collect()
            torch.cuda.empty_cache()

            # ── Evaluate 3D reconstruction ─────────────────────────────────
            if args.eval_3d:
                if gt_depth_maps is None or gt_poses is None or gt_intrinsics is None:
                    print("  WARNING: --eval_3d requires GT depth, poses, and intrinsics. Skipping 3D eval.")
                else:
                    print(f"  Computing 3D reconstruction metrics ...")
                    t0 = time.time()
                    recon_results = evaluate_3d_reconstruction(
                        pred_pts3d_list,
                        gt_depth_maps,
                        gt_intrinsics,
                        gt_poses,
                        pred_poses=pred_poses_list if pred_poses_list else None,
                        icp_threshold=args.icp_threshold,
                        use_umeyama=args.use_umeyama,
                        crop_center=args.crop_center,
                    )
                    scene_results.update({f"recon_{k}": v for k, v in recon_results.items()})
                    print(f"  3D Recon ({time.time() - t0:.1f}s) | Acc: {recon_results['acc']:.4f}, "
                          f"Comp: {recon_results['comp']:.4f}, "
                          f"Chamfer: {recon_results['chamfer']:.4f}, NC: {recon_results['nc']:.4f}")

            # ── Evaluate camera pose (evo-based) ──────────────────────────
            if args.eval_pose:
                if not has_poses or not pred_poses_list:
                    print(f"  WARNING: No predicted poses available. Skipping pose eval.")
                elif gt_poses is None:
                    print("  WARNING: --eval_pose requires GT poses. Skipping pose eval.")
                else:
                    n_common = min(len(pred_poses_list), len(gt_poses))
                    pose_results = eval_pose_metrics(
                        pred_poses_list[:n_common], gt_poses[:n_common]
                    )
                    scene_results.update({f"pose_{k}": v for k, v in pose_results.items()})
                    print(f"  Pose     | ATE: {pose_results['ate']:.4f}, "
                          f"RPE trans: {pose_results['rpe_trans']:.4f}, RPE rot: {pose_results['rpe_rot']:.4f}")
                    print(f"  Pose (per-frame) | ATE frames: {len(pose_results['ate_per_frame'])}, "
                          f"RPE trans frames: {len(pose_results['rpe_trans_per_frame'])}, "
                          f"RPE rot frames: {len(pose_results['rpe_rot_per_frame'])}")

            # ── Evaluate relative pose (pairwise, no evo) ─────────────────
            if args.eval_relpose:
                if not has_poses or not pred_poses_list:
                    print(f"  WARNING: No predicted poses available. Skipping relpose eval.")
                elif gt_poses is None:
                    print("  WARNING: --eval_relpose requires GT poses. Skipping relpose eval.")
                else:
                    n_common = min(len(pred_poses_list), len(gt_poses))
                    pred_se3 = np.stack(pred_poses_list[:n_common])
                    gt_se3 = np.stack(gt_poses[:n_common])
                    rpe_results = relative_pose_error(pred_se3, gt_se3)
                    scene_results["relpose_auc"] = rpe_results["auc"]
                    scene_results["relpose_rot_median"] = float(np.median(rpe_results["rot_errors"]))
                    scene_results["relpose_trans_median"] = float(np.median(rpe_results["trans_errors"]))
                    print(f"  RelPose  | AUC@30: {rpe_results['auc']:.4f}, "
                          f"Rot Med: {scene_results['relpose_rot_median']:.2f}°, "
                          f"Trans Med: {scene_results['relpose_trans_median']:.2f}°")

            # ── Evaluate multi-view depth ─────────────────────────────────
            if args.eval_depth:
                if gt_depth_maps is None:
                    print("  WARNING: --eval_depth requires GT depth maps. Skipping depth eval.")
                elif not pred_pts3d_cam_list and (not has_poses or not pred_poses_list):
                    print("  WARNING: --eval_depth requires pts3d_cam or predicted poses. Skipping depth eval.")
                else:
                    use_pts3d_cam = len(pred_pts3d_cam_list) > 0
                    if use_pts3d_cam:
                        print(f"  Computing multi-view depth metrics using pts3d_cam (align={args.depth_align}) ...")
                    else:
                        print(f"  Computing multi-view depth metrics using pts3d+pose (align={args.depth_align}) ...")
                    t0 = time.time()

                    # Determine max_depth for this dataset
                    max_depth = args.depth_max_depth
                    if max_depth is None:
                        # Dataset-dependent defaults (matching reference eval_depth.py)
                        if args.dataset in ("nrgbd", "7scenes"):
                            max_depth = 10.0
                        elif args.dataset in ("sintel", "bonn", "bonn_full"):
                            max_depth = 70.0
                        elif args.dataset == "kitti":
                            max_depth = None  # KITTI uses no max_depth cap
                        else:
                            max_depth = 80.0

                    # Dataset-specific post_clip_max (sintel clips aligned depth to 70m)
                    post_clip_max = None
                    if args.dataset == "sintel":
                        post_clip_max = 70.0

                    if use_pts3d_cam:
                        n_depth_views = min(len(pred_pts3d_cam_list), len(gt_depth_maps))
                    else:
                        n_depth_views = min(len(pred_pts3d_list), len(pred_poses_list), len(gt_depth_maps))
                    pred_depths = []
                    gt_depths = []

                    for i in range(n_depth_views):
                        # Extract predicted depth
                        if use_pts3d_cam:
                            # Use camera-frame pts3d directly (z-component = depth)
                            pred_depth_i = pred_pts3d_cam_list[i][:, :, 2]
                        else:
                            # Fallback: transform world pts3d to camera frame via pose
                            pred_depth_i = extract_depth_from_pts3d(
                                pred_pts3d_list[i], pred_poses_list[i]
                            )
                        gt_depth_i = gt_depth_maps[i]
                        gt_H, gt_W = gt_depth_i.shape[:2]
                        pred_H, pred_W = pred_depth_i.shape[:2]

                        # Resize predicted depth to GT resolution
                        if (pred_H, pred_W) != (gt_H, gt_W):
                            pred_depth_i = cv2.resize(
                                pred_depth_i.astype(np.float32),
                                (gt_W, gt_H),
                                interpolation=cv2.INTER_CUBIC,
                            )

                        pred_depths.append(pred_depth_i)
                        gt_depths.append(gt_depth_i)

                    pred_depths = np.stack(pred_depths, axis=0)  # (N, H, W)
                    gt_depths = np.stack(gt_depths, axis=0)      # (N, H, W)

                    # Per-sequence alignment (matches Pi3/recons_eval official protocol)
                    if args.depth_align == "scale&shift":
                        depth_results = depth_evaluation(
                            pred_depths, gt_depths,
                            max_depth=max_depth, post_clip_max=post_clip_max,
                            align_with_lad2=True, use_gpu=True,
                        )
                    elif args.depth_align == "scale":
                        depth_results = depth_evaluation(
                            pred_depths, gt_depths,
                            max_depth=max_depth, post_clip_max=post_clip_max,
                            align_with_scale=True, use_gpu=True,
                        )
                    elif args.depth_align == "metric":
                        depth_results = depth_evaluation(
                            pred_depths, gt_depths,
                            max_depth=max_depth, post_clip_max=post_clip_max,
                            metric_scale=True, use_gpu=True,
                        )

                    # Store per-scene depth metrics
                    for k, v in depth_results.items():
                        if k != "valid_pixels":
                            scene_results[f"depth_{k}"] = v
                    scene_results["depth_valid_pixels"] = depth_results["valid_pixels"]

                    print(f"  Depth ({time.time() - t0:.1f}s) | "
                          f"Abs Rel: {depth_results['Abs Rel']:.4f}, "
                          f"RMSE: {depth_results['RMSE']:.4f}, "
                          f"delta<1.25: {depth_results['delta < 1.25']:.4f}")

            # ── Save point clouds ──────────────────────────────────────────
            if args.save_pointclouds:
                scene_out = os.path.join(output_dir, scene_name.replace("/", "_"))
                os.makedirs(scene_out, exist_ok=True)

                all_pts = np.concatenate([p[m] for p, m in zip(pred_pts3d_list, pred_masks_list)], axis=0)
                finite = np.isfinite(all_pts).all(axis=1)
                all_pts = all_pts[finite]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                o3d.io.write_point_cloud(os.path.join(scene_out, "pred.ply"), pcd)
                print(f"  Saved pred point cloud to {scene_out}/pred.ply")

                if gt_depth_maps is not None and gt_poses is not None and gt_intrinsics is not None:
                    all_gt_pts = []
                    for dm, pose in zip(gt_depth_maps, gt_poses):
                        valid = dm > 0
                        gt_pts = backproject_depth(dm, gt_intrinsics, pose)
                        all_gt_pts.append(gt_pts[valid])
                    all_gt_pts = np.concatenate(all_gt_pts, axis=0)
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(all_gt_pts)
                    o3d.io.write_point_cloud(os.path.join(scene_out, "gt.ply"), pcd_gt)
                    print(f"  Saved GT point cloud to {scene_out}/gt.ply")

            # ── Save PLYs in recons_eval format ────────────────────────────
            if args.save_ply_recons_eval:
                if gt_depth_maps is None or gt_poses is None or gt_intrinsics is None:
                    print("  WARNING: --save_ply_recons_eval requires GT depth+pose+intrinsics. Skipping.")
                else:
                    n_views = min(len(pred_pts3d_list), len(gt_depth_maps), len(gt_poses), len(image_paths))
                    pred_stack, gt_stack, valid_stack, color_stack = [], [], [], []
                    for i in range(n_views):
                        pred_i = pred_pts3d_list[i]
                        pred_H, pred_W = pred_i.shape[:2]
                        gt_depth = gt_depth_maps[i]
                        gt_H, gt_W = gt_depth.shape[:2]
                        if (gt_H, gt_W) != (pred_H, pred_W):
                            sx, sy = pred_W / gt_W, pred_H / gt_H
                            gt_depth_r = cv2.resize(gt_depth, (pred_W, pred_H), interpolation=cv2.INTER_NEAREST)
                            intr = gt_intrinsics.copy()
                            intr[0, 0] *= sx
                            intr[1, 1] *= sy
                            intr[0, 2] *= sx
                            intr[1, 2] *= sy
                        else:
                            gt_depth_r = gt_depth
                            intr = gt_intrinsics
                        gt_pts_i = backproject_depth(gt_depth_r, intr, gt_poses[i])
                        img_bgr = cv2.imread(image_paths[i])
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        if img_rgb.shape[:2] != (pred_H, pred_W):
                            img_rgb = cv2.resize(img_rgb, (pred_W, pred_H), interpolation=cv2.INTER_AREA)
                        pred_stack.append(pred_i)
                        gt_stack.append(gt_pts_i)
                        valid_stack.append(gt_depth_r > 0)
                        color_stack.append(img_rgb.astype(np.float32) / 255.0)

                    pred_arr = np.stack(pred_stack, axis=0)
                    gt_arr = np.stack(gt_stack, axis=0)
                    valid_arr = np.stack(valid_stack, axis=0)
                    color_arr = np.stack(color_stack, axis=0)

                    pred_v = pred_arr[valid_arr]
                    gt_v = gt_arr[valid_arr]
                    color_v = color_arr[valid_arr]

                    finite = np.isfinite(pred_v).all(axis=1) & np.isfinite(gt_v).all(axis=1)
                    pred_v = pred_v[finite]
                    gt_v = gt_v[finite]
                    color_v = color_v[finite]

                    s, R_al, t_al = umeyama_alignment(pred_v, gt_v, with_scale=True)
                    pred_aligned = (s * (R_al @ pred_v.T)).T + t_al

                    safe_name = scene_name.replace("/", "_")
                    pred_path = os.path.join(output_dir, f"{safe_name}-pred.ply")
                    gt_path = os.path.join(output_dir, f"{safe_name}-gt.ply")

                    pcd_p = o3d.geometry.PointCloud()
                    pcd_p.points = o3d.utility.Vector3dVector(pred_aligned)
                    pcd_p.colors = o3d.utility.Vector3dVector(color_v)
                    o3d.io.write_point_cloud(pred_path, pcd_p)

                    pcd_g = o3d.geometry.PointCloud()
                    pcd_g.points = o3d.utility.Vector3dVector(gt_v)
                    pcd_g.colors = o3d.utility.Vector3dVector(color_v)
                    o3d.io.write_point_cloud(gt_path, pcd_g)

                    print(f"  [save_ply_recons_eval] umeyama s={s:.4f}, {len(pred_v)} pts → "
                          f"{pred_path}, {gt_path}")

            all_results[scene_name] = scene_results

            # Collect attention diagnostics summary for cross-scene averaging
            if getattr(args, "diagnose_attention", False) and hasattr(args, "_attention_diag"):
                rows = args._attention_diag.get_summary_rows()
                if rows:
                    all_attn_summaries[scene_name] = rows
                # Free the diag object to save memory
                del args._attention_diag

        except Exception as e:
            if "out of memory" in str(e).lower():
                _clear_model_caches(model)
                gc.collect()
                torch.cuda.empty_cache()
                reason = "OOM"
                skipped_scenes[scene_name] = reason
                print(f"  OOM error, skipping scene {scene_name}")
            elif "Skipping scene" in str(e):
                reason = str(e)
                skipped_scenes[scene_name] = reason
                print(f"  Skipping scene {scene_name}: {e}")
            else:
                print(f"  Error in scene {scene_name}: {e}")
                raise

    # ── Aggregate and save results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    metric_keys = set()
    for res in all_results.values():
        for k, v in res.items():
            if isinstance(v, (int, float)) and k not in ("num_views",):
                metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_results.values() if key in r and np.isfinite(r[key])]
        if values:
            # For depth metrics, use weighted average by valid_pixels (matching reference eval)
            if key.startswith("depth_") and key != "depth_valid_pixels":
                weights = [r.get("depth_valid_pixels", 1) for r in all_results.values() if key in r and np.isfinite(r[key])]
                weighted_mean = float(np.average(values, weights=weights))
                summary[key] = {"mean": weighted_mean, "std": float(np.std(values)), "n": len(values), "aggregation": "weighted_by_valid_pixels"}
                print(f"  {key:30s}: {weighted_mean:.5f} ± {np.std(values):.5f}  (n={len(values)}, weighted)")
            else:
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
                print(f"  {key:30s}: {summary[key]['mean']:.5f} ± {summary[key]['std']:.5f}  (n={len(values)})")

    if skipped_scenes:
        print(f"\nSkipped {len(skipped_scenes)} scene(s):")
        for sname, reason in sorted(skipped_scenes.items()):
            print(f"  {sname}: {reason}")

    output = {
        "args": vars(args),
        "per_scene": all_results,
        "skipped_scenes": skipped_scenes,
        "summary": summary,
    }
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    log_path = os.path.join(output_dir, "evaluation_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Num scenes evaluated: {len(all_results)}\n")
        f.write(f"Num scenes skipped: {len(skipped_scenes)}\n")
        if skipped_scenes:
            f.write("Skipped scenes:\n")
            for sname, reason in sorted(skipped_scenes.items()):
                f.write(f"  {sname}: {reason}\n")
        f.write("\n")

        f.write("Per-scene results:\n")
        f.write("-" * 80 + "\n")
        for scene_name, res in sorted(all_results.items()):
            f.write(f"\n{scene_name}:\n")
            for k, v in res.items():
                if k != "scene":
                    f.write(f"  {k}: {v}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Aggregate results:\n")
        for key, val in summary.items():
            f.write(f"  {key:30s}: {val['mean']:.5f} ± {val['std']:.5f}  (n={val['n']})\n")

    print(f"Log saved to {log_path}")

    # ── Aggregate attention diagnostics across scenes ─────────────────────
    if all_attn_summaries:
        import csv
        from collections import defaultdict as _defaultdict

        # Build average table: group by (layer, query_frame), average all metrics
        metric_cols = ["ent_ratio", "top1_pct", "top5_pct", "intra_ent",
                       "intra_t1_pct", "peak_on_pct", "peak_off_pct",
                       "int_ent_on", "int_ent_off"]
        # key = (layer, query_frame) -> list of row dicts
        grouped = _defaultdict(list)
        for scene_rows in all_attn_summaries.values():
            for row in scene_rows:
                grouped[(row["layer"], row["query_frame"])].append(row)

        avg_rows = []
        for (layer, qf) in sorted(grouped.keys()):
            rows_for_key = grouped[(layer, qf)]
            avg_row = {"layer": layer, "query_frame": qf}
            for col in metric_cols:
                vals = [r[col] for r in rows_for_key if np.isfinite(r[col])]
                avg_row[col] = float(np.mean(vals)) if vals else float("nan")
            avg_rows.append(avg_row)

        # Print average table
        print(f"\n{'='*100}")
        print(f"ATTENTION DIAGNOSTICS — AVERAGE ACROSS {len(all_attn_summaries)} SCENES")
        print(f"{'='*100}")
        header = (
            f"{'Layer':>5} | {'QFrame':>6} | {'Ent/Max':>8} | "
            f"{'Top1frm':>8} | {'Top5frm':>8} | "
            f"{'IntraE/M':>8} | {'IntraT1':>8} | "
            f"{'PeakOn':>10} | {'PeakOff':>11} | "
            f"{'IntEnt_ON':>9} | {'IntEnt_OFF':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in avg_rows:
            print(
                f"{r['layer']:>5} | {r['query_frame']:>6} | "
                f"{r['ent_ratio']:>8.3f} | "
                f"{r['top1_pct']:>7.1f}% | {r['top5_pct']:>7.1f}% | "
                f"{r['intra_ent']:>8.3f} | {r['intra_t1_pct']:>7.1f}% | "
                f"{r['peak_on_pct']:>9.2f}% | {r['peak_off_pct']:>10.2f}% | "
                f"{r['int_ent_on']:>9.3f} | {r['int_ent_off']:>10.3f}"
            )

        # Save average CSV
        avg_csv_path = os.path.join(output_dir, "attention_summary_average.csv")
        fieldnames = ["layer", "query_frame"] + metric_cols
        with open(avg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(avg_rows)
        print(f"\nAverage attention summary saved to {avg_csv_path}")

        # Save all per-scene summaries in one combined CSV
        combined_csv_path = os.path.join(output_dir, "attention_summary_all_scenes.csv")
        combined_fieldnames = ["scene"] + fieldnames
        with open(combined_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=combined_fieldnames)
            writer.writeheader()
            for sname, srows in sorted(all_attn_summaries.items()):
                for row in srows:
                    writer.writerow({"scene": sname, **row})
        print(f"Combined per-scene attention summary saved to {combined_csv_path}")

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"\nPeak GPU memory allocated: {peak_mem:.2f} GB")
        print(f"Peak GPU memory reserved:  {peak_mem_reserved:.2f} GB")


if __name__ == "__main__":
    main()
