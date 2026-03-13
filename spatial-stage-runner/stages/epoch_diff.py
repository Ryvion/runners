"""Epoch diff stage — compare two point clouds or meshes for change detection."""

import os
import json
import numpy as np

def _load_points(file_path):
    """Load points from PLY, XYZ, or NPY files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        return np.load(file_path)

    points = []
    with open(file_path, 'r', errors='replace') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if ext == '.ply':
                if line == 'end_header':
                    in_data = True
                    continue
                if not in_data:
                    continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(points) if points else np.zeros((0, 3))


def run_epoch_diff(input_dir, output_dir):
    """Compare reference and current geometry for change detection."""

    point_extensions = {'.ply', '.xyz', '.txt', '.npy'}
    cloud_files = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in point_extensions:
                cloud_files.append(os.path.join(root, f))

    if len(cloud_files) < 2:
        return {
            "ok": False,
            "error": f"Need at least 2 point clouds for diff, found {len(cloud_files)}",
        }

    # First file = reference (from_epoch), second = current (to_epoch)
    ref_points = _load_points(cloud_files[0])
    cur_points = _load_points(cloud_files[1])

    if len(ref_points) < 3 or len(cur_points) < 3:
        return {
            "ok": False,
            "error": f"Insufficient points: reference={len(ref_points)}, current={len(cur_points)}",
        }

    # Compute nearest-neighbor distances from current to reference
    # Use brute force for simplicity (works for moderate point counts)
    sample_size = min(2000, len(cur_points))
    indices = np.random.choice(len(cur_points), sample_size, replace=False) if len(cur_points) > sample_size else np.arange(len(cur_points))
    sampled_cur = cur_points[indices]

    # Compute distances
    distances = np.zeros(len(sampled_cur))
    ref_sample = ref_points[np.random.choice(len(ref_points), min(5000, len(ref_points)), replace=False)] if len(ref_points) > 5000 else ref_points

    for i, pt in enumerate(sampled_cur):
        dists = np.linalg.norm(ref_sample - pt, axis=1)
        distances[i] = np.min(dists)

    # Statistics
    mean_dist = float(np.mean(distances))
    median_dist = float(np.median(distances))
    max_dist = float(np.max(distances))
    std_dist = float(np.std(distances))

    # Classify changes
    threshold = median_dist + 2 * std_dist
    if threshold < 0.01:
        threshold = 0.01

    changed_mask = distances > threshold
    changed_count = int(np.sum(changed_mask))
    change_ratio = changed_count / len(distances)

    # Severity
    if change_ratio > 0.3:
        severity = "critical"
    elif change_ratio > 0.1:
        severity = "warning"
    elif change_ratio > 0.02:
        severity = "info"
    else:
        severity = "none"

    # Generate colored diff cloud (change regions in red)
    diff_points = sampled_cur
    colors = np.zeros((len(diff_points), 3), dtype=np.uint8)
    colors[~changed_mask] = [0, 200, 0]   # Green = no change
    colors[changed_mask] = [255, 0, 0]     # Red = change

    diff_ply_path = os.path.join(output_dir, "diff_overlay.ply")
    with open(diff_ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(diff_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(diff_points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {col[0]} {col[1]} {col[2]}\n")

    # Summary
    summary = {
        "reference_points": len(ref_points),
        "current_points": len(cur_points),
        "sampled_points": len(sampled_cur),
        "mean_distance": round(mean_dist, 6),
        "median_distance": round(median_dist, 6),
        "max_distance": round(max_dist, 6),
        "std_distance": round(std_dist, 6),
        "change_threshold": round(threshold, 6),
        "changed_points": changed_count,
        "change_ratio": round(change_ratio, 4),
        "severity": severity,
    }

    summary_path = os.path.join(output_dir, "diff_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "ok": True,
        "artifact_files": ["diff_overlay.ply", "diff_summary.json"],
        "severity": severity,
        "change_ratio": round(change_ratio, 4),
        "summary": summary,
        "metrics": {
            "reference_points": len(ref_points),
            "current_points": len(cur_points),
            "changed_points": changed_count,
            "engine": "epoch_diff_v1",
        },
    }
