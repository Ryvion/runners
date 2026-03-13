"""Capture QC stage — validate image capture quality."""

import os
import json
import hashlib
import struct

def run_capture_qc(input_dir, output_dir):
    """Assess capture quality: blur, overlap estimation, metadata completeness."""

    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                images.append(os.path.join(root, f))

    if not images:
        return {
            "ok": False,
            "error": "no images found in capture set",
            "scores": {},
        }

    # Sharpness: Laplacian variance per image
    sharpness_scores = []
    try:
        import cv2
        import numpy as np
        for img_path in images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # Normalize: >500 is sharp, <50 is blurry
                score = min(1.0, max(0.0, (lap_var - 50) / 450))
                sharpness_scores.append(score)
    except ImportError:
        # Fallback: check file sizes as proxy for image quality
        for img_path in images:
            size = os.path.getsize(img_path)
            score = min(1.0, size / (2 * 1024 * 1024))  # 2MB = score 1.0
            sharpness_scores.append(score)

    sharpness = sum(sharpness_scores) / len(sharpness_scores) if sharpness_scores else 0

    # Overlap estimation: feature matching between consecutive images
    overlap = 0.0
    try:
        import cv2
        if len(images) >= 2:
            orb = cv2.ORB_create(nfeatures=500)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            overlap_scores = []
            for i in range(min(len(images) - 1, 10)):  # Sample up to 10 pairs
                img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    continue
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
                if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
                    overlap_scores.append(0.0)
                    continue
                matches = bf.match(des1, des2)
                match_ratio = len(matches) / max(len(kp1), len(kp2), 1)
                overlap_scores.append(min(1.0, match_ratio * 3))  # Scale up
            if overlap_scores:
                overlap = sum(overlap_scores) / len(overlap_scores)
    except ImportError:
        overlap = 0.5  # Unknown without OpenCV

    # Metadata completeness: check for EXIF-like sidecar files or filename patterns
    metadata_files = 0
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in {'.xml', '.json', '.xmp', '.csv'}:
                metadata_files += 1
    metadata_score = min(1.0, metadata_files / max(len(images) * 0.5, 1))

    # Coverage: based on image count (heuristic)
    # 20+ images is good coverage for a small site
    coverage = min(1.0, len(images) / 20.0)

    # Geo score: check if any GPS-tagged files or GCP files exist
    geo_files = 0
    for root, _, files in os.walk(input_dir):
        for f in files:
            fl = f.lower()
            if 'gcp' in fl or 'gps' in fl or 'rtk' in fl or fl.endswith('.gpx'):
                geo_files += 1
    geo_score = min(1.0, geo_files / max(1, 1))  # Any geo file = 1.0
    if geo_files == 0:
        geo_score = 0.3  # Partial credit for images that may have embedded EXIF GPS

    # Overall score
    weights = [0.2, 0.25, 0.25, 0.15, 0.15]
    scores = [coverage, sharpness, overlap, geo_score, metadata_score]
    overall = sum(w * s for w, s in zip(weights, scores))

    # Acceptance
    if overall >= 0.7:
        acceptance = "accepted"
    elif overall >= 0.4:
        acceptance = "needs_review"
    else:
        acceptance = "rejected"

    # Issues
    issues = []
    if sharpness < 0.5:
        issues.append({"type": "blur", "severity": "warning", "message": f"Average sharpness score {sharpness:.2f} is below threshold"})
    if overlap < 0.3:
        issues.append({"type": "overlap", "severity": "warning", "message": f"Estimated overlap {overlap:.2f} is low — consider more captures"})
    if coverage < 0.5:
        issues.append({"type": "coverage", "severity": "info", "message": f"Only {len(images)} images found — may be insufficient for large sites"})
    if geo_score < 0.5:
        issues.append({"type": "geo", "severity": "warning", "message": "No GCP/RTK reference files found"})

    # Reflight recommendation
    reflight = {}
    if acceptance == "rejected":
        reflight = {
            "recommended": True,
            "reason": "Overall quality too low for reliable processing",
            "suggestions": [i["message"] for i in issues],
        }

    # Write QC summary
    qc_summary = {
        "image_count": len(images),
        "coverage_score": round(coverage, 4),
        "sharpness_score": round(sharpness, 4),
        "overlap_score": round(overlap, 4),
        "geo_score": round(geo_score, 4),
        "metadata_score": round(metadata_score, 4),
        "overall_score": round(overall, 4),
        "acceptance_status": acceptance,
        "issues": issues,
        "reflight": reflight,
    }

    summary_path = os.path.join(output_dir, "qc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(qc_summary, f, indent=2)

    return {
        "ok": True,
        "artifact_files": ["qc_summary.json"],
        "coverage_score": round(coverage, 4),
        "sharpness_score": round(sharpness, 4),
        "overlap_score": round(overlap, 4),
        "geo_score": round(geo_score, 4),
        "metadata_score": round(metadata_score, 4),
        "overall_score": round(overall, 4),
        "acceptance_status": acceptance,
        "issues": issues,
        "reflight": reflight,
        "metrics": {
            "image_count": len(images),
            "engine": "capture_qc_v1",
        },
    }
