"""Video splitting utilities using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path


def get_video_duration(video_path: str | Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def split_video(video_path: str | Path, segment_duration: int = 20) -> list[dict]:
    """Split a video into fixed-duration segments using ffmpeg.

    Args:
        video_path: Path to the input video file.
        segment_duration: Duration of each segment in seconds.

    Returns:
        List of dicts with keys:
            - path: Path to the segment file
            - start: Start time in seconds (relative to original video)
            - duration: Actual duration of this segment
            - index: Segment index (0-based)
    """
    video_path = Path(video_path)
    total_duration = get_video_duration(video_path)

    if total_duration > 120:
        raise ValueError(f"Video is {total_duration:.0f}s — max supported is 120s")

    tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_segments_"))
    pattern = str(tmp_dir / "segment_%03d.mp4")

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-y", pattern,
        ],
        capture_output=True, check=True,
    )

    segments = []
    for i, seg_file in enumerate(sorted(tmp_dir.glob("segment_*.mp4"))):
        start = i * segment_duration
        seg_duration = min(segment_duration, total_duration - start)
        segments.append({
            "path": seg_file,
            "start": start,
            "duration": seg_duration,
            "index": i,
        })

    return segments
