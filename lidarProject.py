#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)


# -------------------------
# Calibration + I/O helpers
# -------------------------

def _read_vals_after_colon(line):
    """Read floats from a KITTI calib line after the key + colon."""
    return np.array([float(x) for x in line.split()[1:]], dtype=np.float64)


def load_kitti_calib(cam_to_cam_path, velo_to_cam_path, cam_id=2):
    """
    Load KITTI calibration.

    cam_to_cam_path: path to calib_cam_to_cam.txt
    velo_to_cam_path: path to calib_velo_to_cam.txt
    cam_id: which camera (0–3). We use 2 for image_02.
    """
    P_key, R0_key = f'P_rect_0{cam_id}:', 'R_rect_00:'
    P2, R0 = None, None

    with open(cam_to_cam_path, 'r') as f:
        for line in f:
            if line.startswith(P_key):
                P2 = _read_vals_after_colon(line).reshape(3, 4)
            elif line.startswith(R0_key):
                R0 = _read_vals_after_colon(line).reshape(3, 3)

    R, t = None, None
    with open(velo_to_cam_path, 'r') as f:
        for line in f:
            if line.startswith('R:'):
                R = _read_vals_after_colon(line).reshape(3, 3)
            elif line.startswith('T:'):
                t = _read_vals_after_colon(line).reshape(3, 1)

    if P2 is None or R0 is None or R is None or t is None:
        raise RuntimeError("Missing keys in calib files (check paths / file contents).")

    # make them 4x4
    R0_4 = np.eye(4, dtype=np.float64)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4, dtype=np.float64)
    Tr_4[:3, :4] = np.hstack([R, t])  # 3x3 | 3x1  → 3x4

    return P2, R0_4, Tr_4


def load_kitti_bin_xyz(path):
    """Load a KITTI velodyne .bin file and return (N,3) xyz."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # x,y,z


def load_rgb(path):
    """Load an RGB image using OpenCV (BGR→RGB)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -------------------------
# Projection + depth maps
# -------------------------

def project_lidar_to_image(xyz_lidar, P2, R0_4, Tr_4, H, W):
    """
    Project LiDAR points to image plane.
    Returns u, v (pixel coords) and z (depth in camera frame) for valid points.
    """
    # (N,4) homogeneous
    pts = np.hstack([xyz_lidar, np.ones((xyz_lidar.shape[0], 1), dtype=np.float64)])
    # transform: velo -> cam -> rect
    pts_cam = (R0_4 @ (Tr_4 @ pts.T)).T
    xyz_cam = pts_cam[:, :3]

    z = xyz_cam[:, 2]
    front = z > 0.1  # only points in front

    pts_cam_h = np.hstack([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float64)])
    uvw = (P2 @ pts_cam_h.T).T

    u = (uvw[:, 0] / uvw[:, 2]).astype(np.int32)
    v = (uvw[:, 1] / uvw[:, 2]).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & front
    return u[inside], v[inside], z[inside]


def make_sparse_depth(u, v, z, H, W):
    """
    Create sparse depth map and mask from projected points.
    Keep closest depth if multiple points hit the same pixel.
    """
    depth = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), bool)
    for uu, vv, zz in zip(u, v, z):
        if (not mask[vv, uu]) or (zz < depth[vv, uu]):
            depth[vv, uu] = zz
            mask[vv, uu] = True
    return depth, mask


def nearest_fill_depth(depth, mask):
    """
    Fill missing pixels by nearest neighbor using distance transform.
    """
    inv = (~mask).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    ys, xs = np.where(mask)
    src_map = np.zeros_like(labels, dtype=np.int32)
    src_map[ys, xs] = np.arange(len(xs)) + 1

    src_idx = np.maximum(labels - 1, 0)
    src_y, src_x = ys[src_idx], xs[src_idx]

    out = depth.copy()
    out[~mask] = depth[src_y, src_x][~mask]
    return out


def bilateral_depth(depth, mask, d=50, sigma_color=0.1, sigma_space=20.0):
    """
    Nearest-neighbor fill + bilateral smoothing.
    """
    filled = nearest_fill_depth(depth, mask)
    smoothed = cv2.bilateralFilter(filled, d, sigma_color, sigma_space)
    smoothed[mask] = depth[mask]  # keep original LiDAR pixels
    return smoothed


def bilinear_fill_depth(depth, mask):
    """
    Approximate bilinear interpolation of sparse depth via resize trick.
    """
    h, w = depth.shape

    depth_nan = depth.copy()
    depth_nan[~mask] = np.nan

    # avoid zero-size when image is small
    small_w = max(w // 8, 1)
    small_h = max(h // 8, 1)

    small = cv2.resize(depth_nan, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    bilinear = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # restore true LiDAR pixels
    bilinear[mask] = depth[mask]
    return bilinear


def bilinear_plus_bilateral(depth, mask, d=9, sigma_color=0.1, sigma_space=5.0):
    """
    Bilinear interpolation followed by bilateral smoothing.
    """
    bilinear = bilinear_fill_depth(depth, mask)
    smoothed = cv2.bilateralFilter(bilinear, d, sigma_color, sigma_space)
    smoothed[mask] = depth[mask]
    return smoothed


# -------------------------
# Visualization helpers
# -------------------------

def visualize_depth_maps(rgb, sparse, dense_nn, dense_bilinear):
    plt.figure(figsize=(22, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(sparse, cmap="magma")
    plt.title("Sparse Depth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(dense_nn, cmap="magma")
    plt.title("NN + Bilateral")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(dense_bilinear, cmap="magma")
    plt.title("Bilinear + Bilateral")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_point_cloud(xyz):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               s=1, c=xyz[:, 2], cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D LiDAR Point Cloud')

    ax.set_box_aspect([
        np.ptp(xyz[:, 0]),
        np.ptp(xyz[:, 1]),
        np.ptp(xyz[:, 2])
    ])

    plt.show()


def visualize_projection(rgb, u, v):
    H, W, _ = rgb.shape
    lidar_image = np.zeros((H, W, 3), dtype=np.uint8)

    for ui, vi in zip(u, v):
        cv2.circle(lidar_image, (int(ui), int(vi)), 1, (255, 255, 255), -1)

    rgb_with_lidar = rgb.copy()
    for ui, vi in zip(u, v):
        cv2.circle(rgb_with_lidar, (int(ui), int(vi)), 1, (255, 0, 0), -1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lidar_image)
    plt.title("Projected LiDAR on blank image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_with_lidar)
    plt.title("Projected LiDAR on RGB image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Main script
# -------------------------

def main():
    # TODO: CHANGE THESE TO YOUR LOCAL PATHS
    # Root of the drive sequence (contains image_02, velodyne_points, etc.)
    BASE = os.path.expanduser("~/Downloads/2011_09_26_sync/2011_09_26_drive_0001_sync")
    # Folder with calibration txts (calib_cam_to_cam.txt, calib_velo_to_cam.txt)
    CALIB_DIR = os.path.expanduser("~/Downloads/2011_09_26")

    IMG_DIR = os.path.join(BASE, "image_02", "data")
    LIDAR_DIR = os.path.join(BASE, "velodyne_points", "data")

    # Where to save output dense depth maps
    OUT_DIR = os.path.expanduser("~/Downloads/output")
    os.makedirs(OUT_DIR, exist_ok=True)

    cam_to_cam_path = os.path.join(CALIB_DIR, "calib_cam_to_cam.txt")
    velo_to_cam_path = os.path.join(CALIB_DIR, "calib_velo_to_cam.txt")

    P2, R0_4, Tr_4 = load_kitti_calib(
        cam_to_cam_path,
        velo_to_cam_path,
        cam_id=2
    )

    # ---------- Single frame demo ----------
    idx = "0000000030"  # change to any frame index you want
    img_path = os.path.join(IMG_DIR, f"{idx}.png")
    bin_path = os.path.join(LIDAR_DIR, f"{idx}.bin")

    rgb = load_rgb(img_path)
    xyz = load_kitti_bin_xyz(bin_path)
    H, W = rgb.shape[:2]

    u, v, z = project_lidar_to_image(xyz, P2, R0_4, Tr_4, H, W)
    sparse, mask = make_sparse_depth(u, v, z, H, W)

    dense_nn = bilateral_depth(sparse, mask, d=9, sigma_color=0.1, sigma_space=5.0)
    dense_bilinear = bilinear_plus_bilateral(sparse, mask, d=9,
                                             sigma_color=0.1, sigma_space=5.0)

    visualize_depth_maps(rgb, sparse, dense_nn, dense_bilinear)
    visualize_point_cloud(xyz)
    visualize_projection(rgb, u, v)

    # ---------- Optional: process all frames and save dense maps ----------
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    for i, img_path in enumerate(img_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        bin_path = os.path.join(LIDAR_DIR, f"{stem}.bin")

        if not os.path.exists(bin_path):
            print(f"[skip] missing LiDAR for {stem}")
            continue

        rgb = load_rgb(img_path)
        H, W = rgb.shape[:2]
        xyz = load_kitti_bin_xyz(bin_path)

        u, v, z = project_lidar_to_image(xyz, P2, R0_4, Tr_4, H, W)
        sparse, mask = make_sparse_depth(u, v, z, H, W)
        dense = bilateral_depth(sparse, mask, d=9,
                                sigma_color=0.1, sigma_space=5.0)

        enc = np.clip(dense * 256.0, 0, 65535).astype(np.uint16)
        out_path = os.path.join(OUT_DIR, f"{stem}.png")
        cv2.imwrite(out_path, enc)

        if i % 20 == 0:
            print(f"processed {i + 1}/{len(img_paths)}")
    print("done.")


if __name__ == "__main__":
    main()
