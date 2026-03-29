"""Persistence layer — save/load analysis results, checkpointing, and index."""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")
INDEX_PATH = RESULTS_DIR / "index.json"


def _atomic_json_write(path: Path, data) -> None:
    """Write JSON atomically (write to .tmp then rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def compute_input_hash(content: bytes | str) -> str:
    """Return first 8 chars of SHA-256 hash of input content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:8]


def generate_analysis_id(input_type: str, input_hash: str) -> str:
    """Generate a unique analysis ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{input_type}_{input_hash}"


def find_existing_analysis(input_type: str, input_hash: str) -> Path | None:
    """Find the most recent analysis matching this input type and hash.

    Returns the analysis directory path, or None.
    """
    index = load_index()
    # Search newest first
    for entry in reversed(index):
        if entry.get("input_hash") == input_hash and entry.get("input_type") == input_type:
            analysis_dir = RESULTS_DIR / entry["analysis_id"]
            if analysis_dir.exists():
                return analysis_dir
    return None


def init_analysis(
    analysis_id: str,
    input_type: str,
    duration: float,
    label: str,
    input_hash: str,
    n_segments: int = 0,
) -> Path:
    """Create directory structure and write initial metadata."""
    analysis_dir = RESULTS_DIR / analysis_id
    for subdir in ["segments", "heatmaps", "peaks", "gifs", "videos"]:
        (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = {
        "analysis_id": analysis_id,
        "input_type": input_type,
        "label": label,
        "input_hash": input_hash,
        "duration_seconds": round(duration, 1),
        "n_segments": n_segments,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "status": "in_progress",
    }
    _atomic_json_write(analysis_dir / "metadata.json", metadata)

    checkpoint = {
        "status": "in_progress",
        "stages": {
            "split": {"completed": False},
            "tribe_predict": {"completed": False, "segments_done": []},
            "derive": {"completed": False, "segments_done": []},
            "visualize": {"completed": False, "segments_done": []},
            "json_summary": {"completed": False},
            "llm_report": {"completed": False},
            "package": {"completed": False},
        },
    }
    _atomic_json_write(analysis_dir / "checkpoint.json", checkpoint)

    return analysis_dir


def load_checkpoint(analysis_dir: Path) -> dict:
    cp_path = analysis_dir / "checkpoint.json"
    if cp_path.exists():
        with open(cp_path) as f:
            return json.load(f)
    return {"status": "unknown", "stages": {}}


def update_checkpoint(
    analysis_dir: Path,
    stage: str,
    segment_index: int | None = None,
    completed: bool = False,
) -> None:
    """Update checkpoint for a stage, optionally for a specific segment."""
    cp = load_checkpoint(analysis_dir)
    stages = cp.setdefault("stages", {})
    stage_data = stages.setdefault(stage, {"completed": False})

    if segment_index is not None:
        done = stage_data.setdefault("segments_done", [])
        if segment_index not in done:
            done.append(segment_index)
            done.sort()

    if completed:
        stage_data["completed"] = True

    stage_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    _atomic_json_write(analysis_dir / "checkpoint.json", cp)


# --- Segment predictions ---

def save_segment_predictions(
    analysis_dir: Path,
    seg_index: int,
    preds: np.ndarray,
    segments: list,
    events,
) -> None:
    """Save TRIBE v2 predictions for one segment."""
    seg_dir = analysis_dir / "segments"
    np.save(seg_dir / f"seg_{seg_index}_preds.npy", preds)
    with open(seg_dir / f"seg_{seg_index}_segments.pkl", "wb") as f:
        pickle.dump(segments, f)
    events.to_pickle(seg_dir / f"seg_{seg_index}_events.pkl")
    logger.info(f"Saved predictions for segment {seg_index}")


def load_segment_predictions(analysis_dir: Path, seg_index: int) -> dict | None:
    """Load cached predictions. Returns dict with preds/segments/events or None."""
    seg_dir = analysis_dir / "segments"
    preds_path = seg_dir / f"seg_{seg_index}_preds.npy"
    segs_path = seg_dir / f"seg_{seg_index}_segments.pkl"
    events_path = seg_dir / f"seg_{seg_index}_events.pkl"

    if not all(p.exists() for p in [preds_path, segs_path, events_path]):
        return None

    try:
        import pandas as pd
        preds = np.load(preds_path)
        with open(segs_path, "rb") as f:
            segments = pickle.load(f)
        events = pd.read_pickle(events_path)
        return {"preds": preds, "segments": segments, "events": events}
    except Exception as e:
        logger.warning(f"Failed to load cached segment {seg_index}: {e}")
        return None


# --- Derived data (network activations, peaks/drops) ---

def save_segment_derived(
    analysis_dir: Path,
    seg_index: int,
    network_activations: list[dict[str, float]],
    peaks_drops: dict,
) -> None:
    seg_dir = analysis_dir / "segments"
    _atomic_json_write(
        seg_dir / f"seg_{seg_index}_network_activations.json",
        network_activations,
    )
    _atomic_json_write(
        seg_dir / f"seg_{seg_index}_peaks_drops.json",
        peaks_drops,
    )


def load_segment_derived(analysis_dir: Path, seg_index: int) -> dict | None:
    seg_dir = analysis_dir / "segments"
    na_path = seg_dir / f"seg_{seg_index}_network_activations.json"
    pd_path = seg_dir / f"seg_{seg_index}_peaks_drops.json"

    if not na_path.exists() or not pd_path.exists():
        return None

    try:
        with open(na_path) as f:
            network_activations = json.load(f)
        with open(pd_path) as f:
            peaks_drops = json.load(f)
        return {
            "network_activations": network_activations,
            "peaks_drops": peaks_drops,
        }
    except Exception as e:
        logger.warning(f"Failed to load derived data for segment {seg_index}: {e}")
        return None


# --- LLM report ---

def save_report_markdown(analysis_dir: Path, markdown: str) -> None:
    (analysis_dir / "report.md").write_text(markdown)
    logger.info("Saved LLM report markdown")


def load_report_markdown(analysis_dir: Path) -> str | None:
    md_path = analysis_dir / "report.md"
    if md_path.exists():
        return md_path.read_text()
    return None


# --- Completion ---

def mark_complete(analysis_dir: Path) -> None:
    """Mark analysis as complete in checkpoint and metadata."""
    cp = load_checkpoint(analysis_dir)
    cp["status"] = "complete"
    _atomic_json_write(analysis_dir / "checkpoint.json", cp)

    meta_path = analysis_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "complete"
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json_write(meta_path, meta)


# --- Index ---

def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        with open(INDEX_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return []


def update_index(analysis_id: str, metadata: dict) -> None:
    """Add or update an entry in the master index."""
    index = load_index()

    # Update existing or append
    found = False
    for i, entry in enumerate(index):
        if entry.get("analysis_id") == analysis_id:
            index[i] = metadata
            found = True
            break
    if not found:
        index.append(metadata)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(INDEX_PATH, index)


def delete_analysis(analysis_id: str) -> bool:
    """Delete an analysis directory and remove from index."""
    import shutil
    analysis_dir = RESULTS_DIR / analysis_id
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)

    index = load_index()
    index = [e for e in index if e.get("analysis_id") != analysis_id]
    _atomic_json_write(INDEX_PATH, index)
    return True


def list_analysis_files(analysis_dir: Path) -> dict:
    """Collect all output file paths from a completed analysis directory.

    Returns dict with keys matching _run_pipeline outputs:
        mp4_paths, heatmap_gallery, peak_gallery, report_markdown,
        html_path, pdf_path, zip_path
    """
    # MP4s
    mp4_dir = analysis_dir / "videos"
    mp4_paths = sorted(mp4_dir.glob("segment_*.mp4")) if mp4_dir.exists() else []
    mp4_paths = [str(p) for p in mp4_paths]

    # Heatmaps
    hm_dirs = sorted(analysis_dir.glob("heatmaps_seg*"))
    # Also check flat heatmaps/ dir
    if not hm_dirs and (analysis_dir / "heatmaps").exists():
        hm_dirs = [analysis_dir / "heatmaps"]
    heatmap_paths = []
    for d in hm_dirs:
        heatmap_paths.extend(sorted(d.glob("*.png")))
    # Also check flat heatmaps/ dir for files saved there
    if (analysis_dir / "heatmaps").exists():
        for p in sorted((analysis_dir / "heatmaps").glob("*.png")):
            if p not in heatmap_paths:
                heatmap_paths.append(p)

    heatmap_gallery = []
    import re
    for p in heatmap_paths:
        fname = p.stem
        time_match = re.search(r"(\d+\.?\d*)s", fname)
        if time_match:
            secs = float(time_match.group(1))
            label = f"{int(secs // 60)}:{int(secs % 60):02d}"
        else:
            label = "Heatmap"
        heatmap_gallery.append((str(p), label))

    # Peak/drop images
    peak_dirs = sorted(analysis_dir.glob("peaks_seg*")) + sorted(analysis_dir.glob("drops_seg*"))
    if not peak_dirs and (analysis_dir / "peaks").exists():
        peak_dirs = [analysis_dir / "peaks"]
    peak_paths = []
    for d in peak_dirs:
        peak_paths.extend(sorted(d.glob("*.png")))
    if (analysis_dir / "peaks").exists():
        for p in sorted((analysis_dir / "peaks").glob("*.png")):
            if p not in peak_paths:
                peak_paths.append(p)

    peak_gallery = []
    for p in peak_paths:
        fname = p.stem
        time_match = re.search(r"(\d+\.?\d*)s", fname)
        if time_match:
            secs = float(time_match.group(1))
            label = f"{int(secs // 60)}:{int(secs % 60):02d}"
        else:
            label = p.stem
        peak_gallery.append((str(p), label))

    # Report
    report_md = load_report_markdown(analysis_dir)

    # Downloads
    html_path = analysis_dir / "report.html"
    pdf_path = analysis_dir / "report.pdf"
    zip_path = analysis_dir / "tribe_analysis.zip"

    return {
        "mp4_paths": mp4_paths,
        "heatmap_gallery": heatmap_gallery,
        "peak_gallery": peak_gallery,
        "report_markdown": report_md or "",
        "html_path": str(html_path) if html_path.exists() else None,
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "zip_path": str(zip_path) if zip_path.exists() else None,
    }
