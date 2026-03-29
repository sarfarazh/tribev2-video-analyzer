"""Brain heatmap image, GIF, and MP4 generation."""

import io
import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from tribev2.plotting import PlotBrain


def render_single_timestep(
    plotter: PlotBrain,
    preds: np.ndarray,
    segments: list,
    timestep: int,
) -> np.ndarray:
    """Render a single timestep brain heatmap and return as RGB array.

    Args:
        plotter: PlotBrain instance.
        preds: Full predictions array (n_timesteps, n_vertices).
        segments: Full segments list from model.predict().
        timestep: Index of the timestep to render.

    Returns:
        RGB image as numpy array (H, W, 3).
    """
    fig = plotter.plot_timesteps(
        preds[timestep:timestep + 1],
        segments=segments[timestep:timestep + 1],
        cmap="fire",
        norm_percentile=99,
        vmin=0.5,
        alpha_cmap=(0, 0.2),
        show_stimuli=True,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    image = iio.imread(buf)
    return image


def _normalize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Resize all frames to match the first frame's dimensions."""
    if not frames:
        return frames
    target_shape = frames[0].shape
    resized = []
    for frame in frames:
        if frame.shape != target_shape:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            frame = np.array(img)
        resized.append(frame)
    return resized


def generate_interval_heatmaps(
    plotter: PlotBrain,
    preds: np.ndarray,
    segments: list,
    time_offset: float,
    interval: int = 5,
    output_dir: Path | None = None,
) -> list[dict]:
    """Generate brain heatmaps at fixed time intervals.

    Args:
        plotter: PlotBrain instance.
        preds: Predictions for one video segment (n_timesteps, n_vertices).
        segments: Segment objects for this video segment.
        time_offset: Start time of this segment in the original video (seconds).
        interval: Seconds between heatmaps.
        output_dir: Directory to save PNGs. Uses temp dir if None.

    Returns:
        List of dicts with keys: path, timestep, time_label, image.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="tribe_heatmaps_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    n_timesteps = preds.shape[0]
    heatmaps = []

    for t in range(0, n_timesteps, interval):
        image = render_single_timestep(plotter, preds, segments, t)
        abs_time = time_offset + t
        time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"

        filename = f"heatmap_{abs_time:05.1f}s.png"
        path = output_dir / filename
        iio.imwrite(path, image)

        heatmaps.append({
            "path": str(path),
            "timestep": t,
            "abs_time": abs_time,
            "time_label": time_label,
            "image": image,
        })

    return heatmaps


def generate_peak_snapshots(
    plotter: PlotBrain,
    preds: np.ndarray,
    segments: list,
    peak_timesteps: list[int],
    time_offset: float,
    output_dir: Path | None = None,
    label_prefix: str = "peak",
) -> list[dict]:
    """Generate brain heatmaps for specific peak/drop timesteps.

    Args:
        plotter: PlotBrain instance.
        preds: Predictions for one video segment.
        segments: Segment objects for this video segment.
        peak_timesteps: List of timestep indices to render.
        time_offset: Start time of this segment in the original video.
        output_dir: Directory to save PNGs.
        label_prefix: Prefix for filenames ("peak" or "drop").

    Returns:
        List of dicts with keys: path, timestep, time_label, image.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="tribe_peaks_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = []
    for i, t in enumerate(peak_timesteps):
        if t >= preds.shape[0]:
            continue
        image = render_single_timestep(plotter, preds, segments, t)
        abs_time = time_offset + t
        time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"

        filename = f"{label_prefix}_{i}_{abs_time:05.1f}s.png"
        path = output_dir / filename
        iio.imwrite(path, image)

        snapshots.append({
            "path": str(path),
            "timestep": t,
            "abs_time": abs_time,
            "time_label": time_label,
            "image": image,
        })

    return snapshots


def generate_segment_gif(
    plotter: PlotBrain,
    preds: np.ndarray,
    segments: list,
    time_offset: float,
    output_path: Path | None = None,
    fps: int = 1,
) -> str:
    """Generate an animated GIF of brain activity for a video segment.

    Args:
        plotter: PlotBrain instance.
        preds: Predictions for one video segment (n_timesteps, n_vertices).
        segments: Segment objects for this video segment.
        time_offset: Start time in the original video (for labeling).
        output_path: Where to save the GIF. Uses temp file if None.
        fps: Frames per second for the animation.

    Returns:
        Path to the generated GIF file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".gif", prefix="tribe_anim_", delete=False,
        )
        output_path = Path(tmp.name)

    frames = []
    for t in range(preds.shape[0]):
        frame = render_single_timestep(plotter, preds, segments, t)
        frames.append(frame)

    frames = _normalize_frames(frames)

    iio.imwrite(
        output_path,
        frames,
        extension=".gif",
        duration=int(1000 / fps),
        loop=0,
    )

    return str(output_path)


def generate_segment_mp4(
    plotter: PlotBrain,
    preds: np.ndarray,
    segments: list,
    segment_video_path: str | Path,
    time_offset: float,
    output_path: Path | None = None,
) -> str:
    """Generate an MP4 of brain activity with original audio for a video segment.

    Each brain heatmap frame is held for 30 frames (1 second at 30fps),
    and the original audio from the segment is muxed in.

    Args:
        plotter: PlotBrain instance.
        preds: Predictions for one video segment (n_timesteps, n_vertices).
        segments: Segment objects for this video segment.
        segment_video_path: Path to the original video segment (for audio extraction).
        time_offset: Start time in the original video.
        output_path: Where to save the MP4. Uses temp file if None.

    Returns:
        Path to the generated MP4 file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="tribe_video_", delete=False,
        )
        output_path = Path(tmp.name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_mp4_frames_"))

    # Render brain heatmap frames
    frames = []
    for t in range(preds.shape[0]):
        frame = render_single_timestep(plotter, preds, segments, t)
        frames.append(frame)

    frames = _normalize_frames(frames)

    # Write each frame 30 times (30fps, 1 brain frame per second)
    frame_idx = 0
    for t, frame in enumerate(frames):
        for dup in range(30):
            frame_path = tmp_dir / f"frame_{frame_idx:06d}.png"
            iio.imwrite(frame_path, frame)
            frame_idx += 1

    # Extract audio from original segment
    audio_path = tmp_dir / "audio.aac"
    subprocess.run(
        [
            "ffmpeg", "-i", str(segment_video_path),
            "-vn", "-acodec", "aac", "-y", str(audio_path),
        ],
        capture_output=True,
    )
    has_audio = audio_path.exists() and audio_path.stat().st_size > 0

    # Encode frames to MP4, mux with audio if available
    frames_pattern = str(tmp_dir / "frame_%06d.png")
    video_only = tmp_dir / "video_only.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-framerate", "30",
            "-i", frames_pattern,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y", str(video_only),
        ],
        capture_output=True, check=True,
    )

    if has_audio:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_only),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                "-y", str(output_path),
            ],
            capture_output=True, check=True,
        )
    else:
        # No audio — just copy video
        import shutil
        shutil.copy2(video_only, output_path)

    return str(output_path)
