"""Export pack stage — package processed outputs into delivery-ready bundles."""

import os
import json
import shutil
import zipfile

SUPPORTED_PROFILES = {
    "ply", "obj", "stl", "glb", "gltf",  # Mesh formats
    "las", "laz", "xyz",                   # Point cloud formats
    "geotiff", "cog", "png", "jpg",        # Raster/image formats
    "3dtiles", "cesium",                   # Tile formats
    "raw",                                  # Pass-through
}

def run_export_pack(input_dir, output_dir):
    """Package artifacts into a delivery-ready export bundle."""

    # Discover input files
    input_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, input_dir)
            input_files.append((full, rel))

    if not input_files:
        return {
            "ok": False,
            "error": "no input files found for export packaging",
        }

    # Read format profile from job spec if available
    format_profile = os.environ.get("RYV_EXPORT_FORMAT", "raw")
    job_path = "/work/job.json"
    if os.path.exists(job_path):
        try:
            with open(job_path) as f:
                job = json.load(f)
            spec = job.get("spec", {})
            if isinstance(spec, str):
                spec = json.loads(spec)
            format_profile = spec.get("format_profile", format_profile)
        except (json.JSONDecodeError, KeyError):
            pass

    # Build manifest
    manifest_entries = []
    exported_files = []

    for full_path, rel_path in input_files:
        ext = os.path.splitext(rel_path)[1].lower().lstrip('.')
        size = os.path.getsize(full_path)

        # Copy to output directory
        dest = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(full_path, dest)
        exported_files.append(rel_path)

        manifest_entries.append({
            "path": rel_path,
            "format": ext,
            "size_bytes": size,
        })

    # Write manifest
    manifest = {
        "format_profile": format_profile,
        "file_count": len(manifest_entries),
        "total_bytes": sum(e["size_bytes"] for e in manifest_entries),
        "files": manifest_entries,
    }

    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    exported_files.append("export_manifest.json")

    return {
        "ok": True,
        "artifact_files": exported_files,
        "format_profile": format_profile,
        "file_count": len(manifest_entries),
        "total_bytes": manifest["total_bytes"],
        "metrics": {
            "file_count": len(manifest_entries),
            "total_bytes": manifest["total_bytes"],
            "format_profile": format_profile,
            "engine": "export_pack_v1",
        },
    }
