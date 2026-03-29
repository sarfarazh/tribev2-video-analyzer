"""Video/audio splitting and text file utilities using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path


def get_media_duration(media_path: str | Path) -> float:
    """Get duration of a video or audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# Alias for backwards compatibility
get_video_duration = get_media_duration


def trim_video(video_path: str | Path, max_duration: int = 120) -> Path:
    """Trim a video to the first max_duration seconds."""
    video_path = Path(video_path)
    tmp = Path(tempfile.mkdtemp(prefix="tribe_trim_")) / "trimmed.mp4"
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-t", str(max_duration),
            "-c", "copy",
            "-y", str(tmp),
        ],
        capture_output=True, check=True,
    )
    return tmp


def trim_audio(audio_path: str | Path, max_duration: int = 120) -> Path:
    """Trim an audio file to the first max_duration seconds."""
    audio_path = Path(audio_path)
    suffix = audio_path.suffix or ".wav"
    tmp = Path(tempfile.mkdtemp(prefix="tribe_trim_")) / f"trimmed{suffix}"
    subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-t", str(max_duration),
            "-c", "copy",
            "-y", str(tmp),
        ],
        capture_output=True, check=True,
    )
    return tmp


def split_video(video_path: str | Path, segment_duration: int = 20) -> list[dict]:
    """Split a video into fixed-duration segments using ffmpeg."""
    video_path = Path(video_path)
    total_duration = get_media_duration(video_path)

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


def split_audio(audio_path: str | Path, segment_duration: int = 20) -> list[dict]:
    """Split an audio file into fixed-duration segments using ffmpeg."""
    audio_path = Path(audio_path)
    total_duration = get_media_duration(audio_path)
    suffix = audio_path.suffix or ".wav"

    tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_audio_segments_"))
    pattern = str(tmp_dir / f"segment_%03d{suffix}")

    subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-y", pattern,
        ],
        capture_output=True, check=True,
    )

    segments = []
    for i, seg_file in enumerate(sorted(tmp_dir.glob(f"segment_*{suffix}"))):
        start = i * segment_duration
        seg_duration = min(segment_duration, total_duration - start)
        segments.append({
            "path": seg_file,
            "start": start,
            "duration": seg_duration,
            "index": i,
        })

    return segments


def text_to_file(text: str) -> Path:
    """Save text to a temporary .txt file for TRIBE v2 processing."""
    tmp = Path(tempfile.mkdtemp(prefix="tribe_text_")) / "input.txt"
    tmp.write_text(text)
    return tmp
