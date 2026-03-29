"""TRIBE Analyzer — Gradio app for brain response video analysis."""

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
from video import split_video, get_video_duration
from visuals import (
    generate_interval_heatmaps,
    generate_peak_snapshots,
    generate_segment_gif,
)
from analysis import build_full_summary, build_segment_summary, save_summary
from report import generate_report, build_html_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

# Load model at startup
load_model(cache_folder=str(CACHE_FOLDER))


def analyze_video(video_file: str, api_key: str, progress=gr.Progress()):
    """Main analysis pipeline — called by the Gradio UI."""

    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")

    if video_file is None:
        raise gr.Error("Please upload a video file.")

    video_path = Path(video_file)
    duration = get_video_duration(video_path)
    if duration > 120:
        raise gr.Error(f"Video is {duration:.0f}s — max supported is 120 seconds.")

    output_dir = Path(tempfile.mkdtemp(prefix="tribe_output_"))
    plotter = get_plotter()
    atlas_plotter = get_atlas_plotter()

    # --- Step 1: Split video ---
    progress(0.05, desc="Splitting video into segments...")
    logger.info("Splitting video into 20s segments...")
    video_segments = split_video(video_path, segment_duration=20)
    logger.info(f"Split into {len(video_segments)} segments")

    # --- Step 2: Process each segment with TRIBE v2 ---
    all_preds = []
    all_segments = []
    all_events = []
    all_network_activations = []
    all_peaks_drops = []

    for i, seg in enumerate(video_segments):
        frac = 0.1 + (i / len(video_segments)) * 0.4
        progress(frac, desc=f"Processing segment {i + 1}/{len(video_segments)} with TRIBE v2...")
        logger.info(f"Processing segment {i + 1}: {seg['path']}")

        result = process_segment(str(seg["path"]))
        all_preds.append(result["preds"])
        all_segments.append(result["segments"])
        all_events.append(result["events"])

        # Aggregate to networks
        net_act = aggregate_to_networks(result["preds"], atlas_plotter)
        all_network_activations.append(net_act)

        pd_result = find_peaks_and_drops(net_act)
        all_peaks_drops.append(pd_result)

    # --- Step 3: Generate visualizations ---
    progress(0.55, desc="Generating brain heatmaps...")
    logger.info("Generating heatmap images...")

    all_heatmaps = []
    all_peak_images = []
    all_gifs = []

    for i, seg in enumerate(video_segments):
        frac = 0.55 + (i / len(video_segments)) * 0.15
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

        # Animated GIF
        gif_path = generate_segment_gif(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"],
            output_path=output_dir / f"animation_seg{i}.gif",
            fps=2,
        )
        all_gifs.append(gif_path)

    # --- Step 4: Build JSON summary ---
    progress(0.75, desc="Building analysis summary...")
    logger.info("Building JSON summary...")

    segment_summaries = []
    for i, seg in enumerate(video_segments):
        seg_summary = build_segment_summary(
            network_activations=all_network_activations[i],
            events_df=all_events[i],
            time_offset=seg["start"],
            segment_index=i,
            peaks_drops=all_peaks_drops[i],
        )
        segment_summaries.append(seg_summary)

    full_summary = build_full_summary(segment_summaries, duration, segment_duration=20)
    json_path = save_summary(full_summary, output_dir / "analysis.json")

    # --- Step 5: Generate LLM report ---
    progress(0.80, desc="Generating report with Claude Opus 4.6...")
    logger.info("Sending data to Claude Opus 4.6 via OpenRouter...")

    heatmap_paths = [h["path"] for h in all_heatmaps]
    peak_paths = [p["path"] for p in all_peak_images]

    report_markdown = generate_report(
        api_key=api_key.strip(),
        summary=full_summary,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
    )

    # --- Step 6: Build downloadable HTML report ---
    progress(0.95, desc="Preparing downloads...")
    html_report = build_html_report(report_markdown, heatmap_paths, peak_paths)
    html_path = output_dir / "report.html"
    html_path.write_text(html_report)

    logger.info("Analysis complete!")

    # Build gallery data: list of (image_path, caption) tuples
    heatmap_gallery = [(h["path"], h["time_label"]) for h in all_heatmaps]
    peak_gallery = [
        (p["path"], f"{p['time_label']}")
        for p in all_peak_images
    ]

    # Pad GIF list to always have 3 entries (for fixed UI slots)
    while len(all_gifs) < 3:
        all_gifs.append(None)

    return (
        all_gifs[0],               # gif_1
        all_gifs[1],               # gif_2
        all_gifs[2],               # gif_3
        heatmap_gallery,           # heatmap_gallery
        peak_gallery,              # peak_gallery
        report_markdown,           # report_markdown
        str(html_path),            # html_download
        json_path,                 # json_download
    )


# --- Gradio UI ---

with gr.Blocks(
    title="TRIBE Analyzer",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# TRIBE Analyzer\n"
        "Analyze your short-form videos using Meta's TRIBE v2 brain encoding model. "
        "Understand what cognitive processes your content triggers and how to improve engagement.\n\n"
        "*Predictions are model-based estimates, not real brain scans.*"
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="sk-or-...",
            scale=2,
        )

    with gr.Row():
        video_input = gr.Video(label="Upload Video (max 120s)")
        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    # --- Outputs ---
    gr.Markdown("## Brain Activity Animations")
    with gr.Row():
        gif_1 = gr.Image(label="Segment 1 (0:00–0:20)", type="filepath")
        gif_2 = gr.Image(label="Segment 2 (0:20–0:40)", type="filepath")
        gif_3 = gr.Image(label="Segment 3 (0:40–1:00)", type="filepath")

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
        html_download = gr.File(label="Download Report (HTML)")
        json_download = gr.File(label="Download Raw Data (JSON)")

    # Wire up the button
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, api_key_input],
        outputs=[
            gif_1, gif_2, gif_3,
            heatmap_gallery, peak_gallery,
            report_output,
            html_download, json_download,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
