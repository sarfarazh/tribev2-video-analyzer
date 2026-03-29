"""Brain heatmap image and GIF generation."""

import io
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
    fps: int = 2,
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

    # Resize all frames to match the first frame's dimensions
    target_shape = frames[0].shape
    resized = []
    for frame in frames:
        if frame.shape != target_shape:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            frame = np.array(img)
        resized.append(frame)

    iio.imwrite(
        output_path,
        resized,
        extension=".gif",
        duration=int(1000 / fps),
        loop=0,
    )

    return str(output_path)
