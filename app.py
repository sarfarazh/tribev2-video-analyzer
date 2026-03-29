"""TRIBE Analyzer — Gradio app with Script, Voiceover, and Video tabs."""

import os

# Must be set before any HuggingFace imports — RunPod sets this to 1
# but hf_transfer is not installed, causing ImportError during model download.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.5"

import json
import logging
import tempfile
from pathlib import Path

import gradio as gr

from brain import (
    aggregate_to_networks,
    find_peaks_and_drops,
    get_atlas_plotter,
    get_plotter,
    load_model,
    process_segment,
)
from video import (
    get_media_duration,
    split_audio,
    split_video,
    text_to_file,
    trim_audio,
    trim_video,
)
from visuals import (
    generate_interval_heatmaps,
    generate_peak_snapshots,
    generate_segment_gif,
    generate_segment_mp4,
)
from analysis import build_full_summary, build_segment_summary, save_summary
from report import (
    generate_report,
    build_html_report,
    build_pdf_report,
    build_zip_package,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

MAX_SEGMENTS = 6  # Max video/audio slots in UI (120s / 20s)

# Load model at startup
load_model(cache_folder=str(CACHE_FOLDER))


# ---------------------------------------------------------------------------
# Shared analysis pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    input_type: str,
    api_key: str,
    media_segments: list[dict],
    total_duration: float,
    progress: gr.Progress,
    progress_offset: float = 0.1,
) -> tuple:
    """Core pipeline shared by all three tabs.

    Args:
        input_type: "video", "audio", or "text".
        api_key: OpenRouter API key.
        media_segments: List of dicts with keys: path, start, duration, index.
        total_duration: Total duration in seconds.
        progress: Gradio progress tracker.
        progress_offset: Where to start the progress bar.

    Returns:
        Tuple of outputs matching the Gradio output components.
    """
    output_dir = Path(tempfile.mkdtemp(prefix="tribe_output_"))
    plotter = get_plotter()
    atlas_plotter = get_atlas_plotter()

    # --- Process each segment with TRIBE v2 ---
    all_preds = []
    all_segments = []
    all_events = []
    all_network_activations = []
    all_peaks_drops = []

    n_segs = len(media_segments)
    for i, seg in enumerate(media_segments):
        frac = progress_offset + (i / n_segs) * 0.4
        progress(frac, desc=f"Processing segment {i + 1}/{n_segs} with TRIBE v2...")
        logger.info(f"Processing segment {i + 1}: {seg['path']}")

        result = process_segment(str(seg["path"]), input_type=input_type)
        all_preds.append(result["preds"])
        all_segments.append(result["segments"])
        all_events.append(result["events"])

        net_act = aggregate_to_networks(result["preds"], atlas_plotter)
        all_network_activations.append(net_act)

        pd_result = find_peaks_and_drops(net_act)
        all_peaks_drops.append(pd_result)

    # --- Generate visualizations ---
    progress(0.55, desc="Generating brain heatmaps...")
    logger.info("Generating heatmap images...")

    all_heatmaps = []
    all_peak_images = []
    all_mp4s = []
    all_gifs = []

    for i, seg in enumerate(media_segments):
        frac = 0.55 + (i / n_segs) * 0.15
        progress(frac, desc=f"Rendering visualizations for segment {i + 1}...")

        # 5-second interval heatmaps
        heatmaps = generate_interval_heatmaps(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"],
            interval=5,
            output_dir=output_dir / f"heatmaps_seg{i}",
        )
        all_heatmaps.extend(heatmaps)

        # Peak/drop snapshots
        peak_indices = [t for t, _ in all_peaks_drops[i]["peaks"]]
        drop_indices = [t for t, _ in all_peaks_drops[i]["drops"]]

        peaks = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=peak_indices,
            time_offset=seg["start"],
            output_dir=output_dir / f"peaks_seg{i}",
            label_prefix="peak",
        )
        drops = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=drop_indices,
            time_offset=seg["start"],
            output_dir=output_dir / f"drops_seg{i}",
            label_prefix="drop",
        )
        all_peak_images.extend(peaks + drops)

        # Animated MP4 (with audio from original segment if video/audio)
        if input_type in ("video", "audio"):
            mp4_path = generate_segment_mp4(
                plotter, all_preds[i], all_segments[i],
                segment_video_path=str(seg["path"]),
                time_offset=seg["start"],
                output_path=output_dir / f"segment_{i}.mp4",
            )
            all_mp4s.append(mp4_path)

        # Animated GIF for HTML report embedding
        gif_path = generate_segment_gif(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"],
            output_path=output_dir / f"segment_{i}.gif",
            fps=1,
        )
        all_gifs.append(gif_path)

    # For text input, generate GIF-only MP4s (no audio to mux)
    if input_type == "text":
        for i, seg in enumerate(media_segments):
            gif_path = all_gifs[i] if i < len(all_gifs) else None
            if gif_path:
                # Convert GIF to MP4 for Gradio video display
                mp4_path = output_dir / f"segment_{i}.mp4"
                import subprocess
                subprocess.run(
                    [
                        "ffmpeg", "-i", str(gif_path),
                        "-movflags", "faststart",
                        "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-y", str(mp4_path),
                    ],
                    capture_output=True,
                )
                if mp4_path.exists():
                    all_mp4s.append(str(mp4_path))

    # --- Build JSON summary ---
    progress(0.75, desc="Building analysis summary...")
    logger.info("Building JSON summary...")

    segment_summaries = []
    for i, seg in enumerate(media_segments):
        seg_summary = build_segment_summary(
            network_activations=all_network_activations[i],
            events_df=all_events[i],
            time_offset=seg["start"],
            segment_index=i,
            peaks_drops=all_peaks_drops[i],
        )
        segment_summaries.append(seg_summary)

    full_summary = build_full_summary(segment_summaries, total_duration, segment_duration=20)
    json_path = save_summary(full_summary, output_dir / "analysis.json")

    # --- Generate LLM report ---
    progress(0.80, desc="Generating report with Claude Opus 4.6...")
    logger.info("Sending data to Claude Opus 4.6 via OpenRouter...")

    heatmap_paths = [h["path"] for h in all_heatmaps]
    peak_paths = [p["path"] for p in all_peak_images]

    report_markdown = generate_report(
        api_key=api_key.strip(),
        summary=full_summary,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
        input_type=input_type,
    )

    # --- Build downloadable reports ---
    progress(0.90, desc="Preparing downloads...")

    html_report = build_html_report(
        report_markdown, heatmap_paths, peak_paths, gif_paths=all_gifs,
    )
    html_path = output_dir / "report.html"
    html_path.write_text(html_report)

    pdf_path = output_dir / "report.pdf"
    try:
        build_pdf_report(str(html_path), str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        pdf_path = None

    progress(0.95, desc="Building download package...")
    zip_path = build_zip_package(
        output_path=output_dir / "tribe_analysis.zip",
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_path and pdf_path.exists() else None,
        json_path=json_path,
        mp4_paths=all_mp4s,
        gif_paths=all_gifs,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
    )

    logger.info("Analysis complete!")

    # Build gallery data
    heatmap_gallery = [(h["path"], h["time_label"]) for h in all_heatmaps]
    peak_gallery = [(p["path"], p["time_label"]) for p in all_peak_images]

    # Pad MP4 list to always have MAX_SEGMENTS entries
    while len(all_mp4s) < MAX_SEGMENTS:
        all_mp4s.append(None)

    return (
        all_mp4s[0],
        all_mp4s[1],
        all_mp4s[2],
        all_mp4s[3],
        all_mp4s[4],
        all_mp4s[5],
        heatmap_gallery,
        peak_gallery,
        report_markdown,
        str(html_path),
        str(pdf_path) if pdf_path and pdf_path.exists() else None,
        zip_path,
    )


# ---------------------------------------------------------------------------
# Tab-specific entry points
# ---------------------------------------------------------------------------

def analyze_script(script_text: str, api_key: str, progress=gr.Progress()):
    """Script tab — text input analyzed via TRIBE v2 gTTS pipeline."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if not script_text or not script_text.strip():
        raise gr.Error("Please enter your script text.")

    progress(0.02, desc="Saving script to file...")
    text_path = text_to_file(script_text.strip())

    # TRIBE v2 handles text internally (gTTS conversion) — single segment, no splitting
    segments = [{
        "path": text_path,
        "start": 0,
        "duration": 0,  # unknown until processed
        "index": 0,
    }]

    return _run_pipeline(
        input_type="text",
        api_key=api_key,
        media_segments=segments,
        total_duration=0,  # will be approximate
        progress=progress,
    )


def analyze_voiceover(audio_file: str, api_key: str, progress=gr.Progress()):
    """Voiceover tab — audio file split and analyzed."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")

    audio_path = Path(audio_file)
    duration = get_media_duration(audio_path)

    if duration > 150:
        raise gr.Error(f"Audio is {duration:.0f}s — max supported is 2:30.")

    if duration > 120:
        progress(0.02, desc="Trimming audio to 120s...")
        logger.info(f"Audio is {duration:.0f}s — trimming to first 120s")
        audio_path = trim_audio(audio_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting audio into segments...")
    logger.info("Splitting audio into 20s segments...")
    audio_segments = split_audio(audio_path, segment_duration=20)
    logger.info(f"Split into {len(audio_segments)} segments")

    return _run_pipeline(
        input_type="audio",
        api_key=api_key,
        media_segments=audio_segments,
        total_duration=duration,
        progress=progress,
    )


def analyze_video(video_file: str, api_key: str, progress=gr.Progress()):
    """Video tab — video file split and analyzed."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    video_path = Path(video_file)
    duration = get_media_duration(video_path)

    if duration > 150:
        raise gr.Error(f"Video is {duration:.0f}s — max supported is 2:30.")

    if duration > 120:
        progress(0.02, desc="Trimming video to 120s...")
        logger.info(f"Video is {duration:.0f}s — trimming to first 120s")
        video_path = trim_video(video_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting video into segments...")
    logger.info("Splitting video into 20s segments...")
    video_segments = split_video(video_path, segment_duration=20)
    logger.info(f"Split into {len(video_segments)} segments")

    return _run_pipeline(
        input_type="video",
        api_key=api_key,
        media_segments=video_segments,
        total_duration=duration,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TRIBE Analyzer",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# TRIBE Analyzer\n"
        "Analyze your content using Meta's TRIBE v2 brain encoding model. "
        "Understand what cognitive processes your content triggers and how to improve engagement.\n\n"
        "*Predictions are model-based estimates, not real brain scans.*"
    )

    api_key_input = gr.Textbox(
        label="OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
    )

    # --- Input tabs ---
    with gr.Tabs():
        with gr.Tab("Script"):
            gr.Markdown(
                "Paste your script below. TRIBE v2 will convert it to speech "
                "internally and predict brain responses to the language."
            )
            script_input = gr.Textbox(
                label="Script Text",
                placeholder="Paste your video script here...",
                lines=10,
            )
            script_btn = gr.Button("Analyze Script", variant="primary", size="lg")

        with gr.Tab("Voiceover"):
            gr.Markdown(
                "Upload your voiceover or audio file. TRIBE v2 will predict "
                "brain responses to the audio content."
            )
            audio_input = gr.Audio(
                label="Upload Audio (max 2:30)",
                type="filepath",
            )
            audio_btn = gr.Button("Analyze Voiceover", variant="primary", size="lg")

        with gr.Tab("Video"):
            gr.Markdown(
                "Upload your video. TRIBE v2 will analyze both the visual and "
                "audio content to predict full brain responses."
            )
            video_input = gr.Video(label="Upload Video (max 2:30)")
            video_btn = gr.Button("Analyze Video", variant="primary", size="lg")

    # --- Shared outputs ---
    gr.Markdown("---")
    gr.Markdown("## Brain Activity Videos")
    with gr.Row():
        video_1 = gr.Video(label="Segment 1 (0:00–0:20)")
        video_2 = gr.Video(label="Segment 2 (0:20–0:40)")
        video_3 = gr.Video(label="Segment 3 (0:40–1:00)")
    with gr.Row():
        video_4 = gr.Video(label="Segment 4 (1:00–1:20)")
        video_5 = gr.Video(label="Segment 5 (1:20–1:40)")
        video_6 = gr.Video(label="Segment 6 (1:40–2:00)")

    gr.Markdown("## 5-Second Interval Heatmaps")
    heatmap_gallery = gr.Gallery(
        label="Brain activity every 5 seconds",
        columns=4,
        height="auto",
    )

    gr.Markdown("## Peak & Drop Moments")
    peak_gallery = gr.Gallery(
        label="Highest activation peaks and lowest drops",
        columns=5,
        height="auto",
    )

    gr.Markdown("## Analysis Report")
    report_output = gr.Markdown(label="Report")

    gr.Markdown("## Downloads")
    with gr.Row():
        html_download = gr.File(label="Report (HTML)")
        pdf_download = gr.File(label="Report (PDF)")
        zip_download = gr.File(label="Download All (ZIP)")

    # --- Shared output list ---
    outputs = [
        video_1, video_2, video_3,
        video_4, video_5, video_6,
        heatmap_gallery, peak_gallery,
        report_output,
        html_download, pdf_download, zip_download,
    ]

    # Wire up all three buttons
    script_btn.click(
        fn=analyze_script,
        inputs=[script_input, api_key_input],
        outputs=outputs,
    )
    audio_btn.click(
        fn=analyze_voiceover,
        inputs=[audio_input, api_key_input],
        outputs=outputs,
    )
    video_btn.click(
        fn=analyze_video,
        inputs=[video_input, api_key_input],
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
