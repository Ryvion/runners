#!/usr/bin/env python3
"""Ryvion spatial pipeline runner.

This stage runner turns the placeholder spatial pipeline into actual geometry work:
- `spatial_recon`: two-view sparse reconstruction from input images
- `pointcloud_align`: point-cloud normalization / coarse alignment
- `mesh_optimize`: convex-hull mesh generation and cleanup
- `scene_render`: preview rendering from mesh or point cloud
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

WORK_DIR = Path("/work")
JOB_PATH = WORK_DIR / "job.json"
OUTPUT_PATH = WORK_DIR / "output"
RECEIPT_PATH = WORK_DIR / "receipt.json"
METRICS_PATH = WORK_DIR / "metrics.json"
INPUT_DIR = WORK_DIR / "_input"
STAGE_KIND = os.environ.get("RYV_STAGE_KIND", "").strip().lower()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm"}
POINT_EXTS = {".ply", ".xyz", ".txt", ".npy"}
MESH_EXTS = {".ply", ".obj", ".stl", ".glb", ".off"}


@dataclass
class StageOutput:
    files: list[Path]
    summary: dict
    metrics: dict


def load_job() -> dict:
    with JOB_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_kind(job: dict) -> str:
    kind = STAGE_KIND
    if not kind or kind == "spatial_stage":
        kind = str(job.get("kind") or job.get("workload") or "").strip().lower()
    aliases = {
        "pointcloud_alignment": "pointcloud_align",
        "mesh_optimization": "mesh_optimize",
        "scene_rendering": "scene_render",
    }
    kind = aliases.get(kind, kind)
    if not kind:
        raise RuntimeError("spatial stage kind missing")
    return kind


def prepare_input_dir(job: dict) -> Path:
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_file = str(job.get("input_file") or "").strip()
    if src_file and Path(src_file).exists():
        copy_or_extract(Path(src_file), INPUT_DIR)

    payload_url = str(job.get("payload_url") or job.get("input_url") or "").strip()
    if payload_url:
        download_target = INPUT_DIR / Path(payload_url).name
        if download_target.suffix == "":
            download_target = INPUT_DIR / "payload.bin"
        with urllib.request.urlopen(payload_url, timeout=300) as response:
            download_target.write_bytes(response.read())
        copy_or_extract(download_target, INPUT_DIR)

    if not any(INPUT_DIR.iterdir()):
        raise RuntimeError("input payload missing")
    return INPUT_DIR


def copy_or_extract(source: Path, destination: Path) -> None:
    if source.is_dir():
        for child in source.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        return

    suffix = source.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(destination)
        return

    target = destination / source.name
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)


def find_files(root: Path, exts: set[str]) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            matches.append(path)
    return sorted(matches)


def ensure_float_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError("point cloud requires Nx3 coordinates")
    points = points[:, :3]
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if len(points) == 0:
        raise RuntimeError("point cloud is empty after filtering")
    return points


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    points = ensure_float_points(points)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def read_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return ensure_float_points(np.load(path))
    if suffix == ".ply":
        header_complete = False
        vertex_count = 0
        points: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not header_complete:
                    if stripped.startswith("element vertex "):
                        vertex_count = int(stripped.split()[-1])
                    if stripped == "end_header":
                        header_complete = True
                    continue
                if not stripped:
                    continue
                coords = stripped.split()
                if len(coords) < 3:
                    continue
                points.append([float(coords[0]), float(coords[1]), float(coords[2])])
                if vertex_count and len(points) >= vertex_count:
                    break
        if points:
            return ensure_float_points(np.asarray(points, dtype=np.float64))
        cloud = trimesh.load(path, force="scene")
        if isinstance(cloud, trimesh.Trimesh):
            return ensure_float_points(cloud.vertices)
        if isinstance(cloud, trimesh.points.PointCloud):
            return ensure_float_points(cloud.vertices)
        if hasattr(cloud, "vertices"):
            return ensure_float_points(np.asarray(cloud.vertices))
    raw = np.loadtxt(path, dtype=np.float64)
    return ensure_float_points(raw)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sparse_reconstruction(images: list[Path], output_dir: Path) -> StageOutput:
    if len(images) < 2:
        raise RuntimeError("spatial_recon requires at least two images")

    img1 = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(images[1]), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise RuntimeError("failed to read input images")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(4000)
    keypoints1, desc1 = orb.detectAndCompute(gray1, None)
    keypoints2, desc2 = orb.detectAndCompute(gray2, None)
    if desc1 is None or desc2 is None:
        raise RuntimeError("failed to extract visual features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidate_matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in candidate_matches if n is not None and m.distance < 0.75 * n.distance]
    if len(good_matches) < 12:
        raise RuntimeError("not enough feature matches for sparse reconstruction")

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    height, width = gray1.shape[:2]
    focal = 0.9 * max(height, width)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        raise RuntimeError("failed to estimate essential matrix")

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    if len(inliers1) < 8:
        raise RuntimeError("not enough inlier matches after RANSAC")

    _, R, t, pose_mask = cv2.recoverPose(E, inliers1, inliers2, K)
    keep = pose_mask.ravel() == 255
    inliers1 = inliers1[keep]
    inliers2 = inliers2[keep]
    if len(inliers1) < 8:
        raise RuntimeError("not enough points after pose recovery")

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    points_h = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
    points = (points_h[:3] / points_h[3]).T
    points = ensure_float_points(points)
    points = points[np.abs(points[:, 2]) < np.percentile(np.abs(points[:, 2]), 95)]
    if len(points) < 20:
        raise RuntimeError("triangulation produced too few stable points")

    points -= points.mean(axis=0)
    scale = np.median(np.linalg.norm(points, axis=1)) or 1.0
    points /= scale

    cloud_path = output_dir / "sparse_pointcloud.ply"
    write_ascii_ply(cloud_path, points)
    summary = {
        "stage_kind": "spatial_recon",
        "input_images": [img.name for img in images[:2]],
        "keypoints_image_1": len(keypoints1),
        "keypoints_image_2": len(keypoints2),
        "matched_features": len(good_matches),
        "triangulated_points": int(len(points)),
    }
    return StageOutput(
        files=[cloud_path],
        summary=summary,
        metrics={
            "engine": "opencv_two_view_recon",
            "matched_features": len(good_matches),
            "triangulated_points": int(len(points)),
        },
    )


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    dst_centered = target - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - (r @ src_centroid)
    return r, t


def align_pointclouds(pointclouds: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    reference = ensure_float_points(pointclouds[0])
    transforms = [{"rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}]
    aligned = [reference]

    for cloud in pointclouds[1:]:
        source = ensure_float_points(cloud)
        if len(source) > 800:
            source = source[np.linspace(0, len(source) - 1, 800, dtype=int)]
        if len(reference) > 800:
            target = reference[np.linspace(0, len(reference) - 1, 800, dtype=int)]
        else:
            target = reference

        total_r = np.eye(3)
        total_t = np.zeros(3)
        current = source.copy()
        for _ in range(6):
            distances = np.sum((current[:, None, :] - target[None, :, :]) ** 2, axis=2)
            nearest = target[np.argmin(distances, axis=1)]
            r, t = best_fit_transform(current, nearest)
            current = (current @ r.T) + t
            total_r = r @ total_r
            total_t = (r @ total_t) + t

        aligned.append(current)
        transforms.append({"rotation": total_r.tolist(), "translation": total_t.tolist()})

    merged = np.concatenate(aligned, axis=0)
    merged -= merged.mean(axis=0)
    return merged, transforms


def pointcloud_alignment(pointcloud_files: list[Path], output_dir: Path) -> StageOutput:
    if not pointcloud_files:
        raise RuntimeError("pointcloud_align requires at least one point cloud")
    clouds = [read_points(path) for path in pointcloud_files]
    merged, transforms = align_pointclouds(clouds)
    merged /= np.max(np.linalg.norm(merged, axis=1)) or 1.0

    cloud_path = output_dir / "aligned_pointcloud.ply"
    write_ascii_ply(cloud_path, merged)
    transform_path = output_dir / "alignment.json"
    transform_path.write_text(json.dumps({"transforms": transforms}, indent=2), encoding="utf-8")
    return StageOutput(
        files=[cloud_path, transform_path],
        summary={
            "stage_kind": "pointcloud_align",
            "input_clouds": [path.name for path in pointcloud_files],
            "merged_points": int(len(merged)),
            "transforms": len(transforms),
        },
        metrics={
            "engine": "numpy_icp",
            "input_clouds": len(pointcloud_files),
            "merged_points": int(len(merged)),
        },
    )


def mesh_optimization(mesh_sources: list[Path], output_dir: Path) -> StageOutput:
    if not mesh_sources:
        raise RuntimeError("mesh_optimize requires point cloud or mesh input")

    source = mesh_sources[0]
    if source.suffix.lower() in {".ply", ".xyz", ".txt", ".npy"}:
        points = read_points(source)
        hull = ConvexHull(points)
        mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices, process=True)
    else:
        loaded = trimesh.load(source, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise RuntimeError("failed to load mesh source")
        mesh = loaded

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    mesh_path = output_dir / "optimized_mesh.ply"
    mesh.export(mesh_path)
    summary_path = output_dir / "mesh_summary.json"
    summary_payload = {
        "stage_kind": "mesh_optimize",
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "source": source.name,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return StageOutput(
        files=[mesh_path, summary_path],
        summary=summary_payload,
        metrics={
            "engine": "trimesh_convex_hull",
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        },
    )


def render_scene(scene_sources: list[Path], output_dir: Path) -> StageOutput:
    if not scene_sources:
        raise RuntimeError("scene_render requires mesh or point cloud input")

    source = scene_sources[0]
    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#050505")
    fig.patch.set_facecolor("#050505")

    vertices: np.ndarray
    faces = None
    if source.suffix.lower() in {".ply", ".obj", ".stl", ".glb", ".off"}:
        loaded = trimesh.load(source, force="mesh")
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            vertices = ensure_float_points(loaded.vertices)
            faces = np.asarray(loaded.faces)
        else:
            vertices = read_points(source)
    else:
        vertices = read_points(source)

    vertices -= vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices, axis=1)) or 1.0
    vertices /= scale

    if faces is not None and len(faces) > 0:
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color="#9be7ff", linewidth=0.08, alpha=0.92)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c="#9be7ff", alpha=0.85)

    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    preview_path = output_dir / "render_preview.png"
    plt.tight_layout(pad=0)
    fig.savefig(preview_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return StageOutput(
        files=[preview_path],
        summary={
            "stage_kind": "scene_render",
            "source": source.name,
            "preview": preview_path.name,
        },
        metrics={
            "engine": "matplotlib_preview",
            "rendered_vertices": int(len(vertices)),
            "has_faces": bool(faces is not None and len(faces) > 0),
        },
    )


def package_stage_output(stage_name: str, result: StageOutput) -> tuple[bytes, dict]:
    bundle = io.BytesIO()
    summary = dict(result.summary)
    summary.setdefault("stage_kind", stage_name)

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(summary, indent=2))
        for path in result.files:
            archive.write(path, arcname=path.name)

    data = bundle.getvalue()
    payload = {
        "output_hash": hash_bytes(data),
        "ok": True,
        "stage_kind": stage_name,
        "artifact_files": [path.name for path in result.files],
    }
    return data, payload


def write_outputs(stage_name: str, result: StageOutput) -> None:
    data, receipt = package_stage_output(stage_name, result)
    OUTPUT_PATH.write_bytes(data)
    RECEIPT_PATH.write_text(json.dumps(receipt), encoding="utf-8")
    result.metrics.setdefault("output_bytes", len(data))
    result.metrics.setdefault("artifact_files", [path.name for path in result.files])
    METRICS_PATH.write_text(json.dumps(result.metrics), encoding="utf-8")
    print(json.dumps(receipt))


def run_stage(stage: str, input_root: Path) -> StageOutput:
    output_dir = Path(tempfile.mkdtemp(prefix=f"ryv_{stage}_"))
    try:
        if stage == "spatial_recon":
            result = sparse_reconstruction(find_files(input_root, IMAGE_EXTS), output_dir)
        elif stage == "pointcloud_align":
            result = pointcloud_alignment(find_files(input_root, POINT_EXTS), output_dir)
        elif stage == "mesh_optimize":
            result = mesh_optimization(find_files(input_root, POINT_EXTS | MESH_EXTS), output_dir)
        elif stage == "scene_render":
            result = render_scene(find_files(input_root, POINT_EXTS | MESH_EXTS), output_dir)
        else:
            raise RuntimeError(f"unsupported spatial stage {stage}")
        return result
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    try:
        job = load_job()
        stage = stage_kind(job)
        input_root = prepare_input_dir(job)
        result = run_stage(stage, input_root)
        write_outputs(stage, result)
        return 0
    except Exception as exc:  # noqa: BLE001
        fallback_stage = STAGE_KIND or "spatial_stage"
        RECEIPT_PATH.write_text(json.dumps({"ok": False, "error": str(exc), "stage_kind": fallback_stage}), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps({"engine": fallback_stage, "error": str(exc)}), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
