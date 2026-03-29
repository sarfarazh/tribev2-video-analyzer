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
    if duration > 150:
        raise gr.Error(f"Video is {duration:.0f}s — max supported is 2:30.")

    # Trim to 120s if longer
    if duration > 120:
        from video import trim_video
        logger.info(f"Video is {duration:.0f}s — trimming to first 120s")
        video_path = trim_video(video_path, max_duration=120)
        duration = 120

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

        net_act = aggregate_to_networks(result["preds"], atlas_plotter)
        all_network_activations.append(net_act)

        pd_result = find_peaks_and_drops(net_act)
        all_peaks_drops.append(pd_result)

    # --- Step 3: Generate visualizations ---
    progress(0.55, desc="Generating brain heatmaps...")
    logger.info("Generating heatmap images...")

    all_heatmaps = []
    all_peak_images = []
    all_mp4s = []
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

        # Animated MP4 (30fps with audio) for Gradio
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

    # --- Step 6: Build downloadable reports ---
    progress(0.90, desc="Preparing downloads...")

    # HTML report (with embedded GIFs)
    html_report = build_html_report(
        report_markdown, heatmap_paths, peak_paths, gif_paths=all_gifs,
    )
    html_path = output_dir / "report.html"
    html_path.write_text(html_report)

    # PDF report
    pdf_path = output_dir / "report.pdf"
    try:
        build_pdf_report(str(html_path), str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        pdf_path = None

    # ZIP package
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
    peak_gallery = [
        (p["path"], p["time_label"])
        for p in all_peak_images
    ]

    # Pad MP4 list to always have 6 entries (max segments for 120s)
    while len(all_mp4s) < 6:
        all_mp4s.append(None)

    return (
        all_mp4s[0],               # video_1
        all_mp4s[1],               # video_2
        all_mp4s[2],               # video_3
        all_mp4s[3],               # video_4
        all_mp4s[4],               # video_5
        all_mp4s[5],               # video_6
        heatmap_gallery,           # heatmap_gallery
        peak_gallery,              # peak_gallery
        report_markdown,           # report_markdown
        str(html_path),            # html_download
        str(pdf_path) if pdf_path and pdf_path.exists() else None,  # pdf_download
        zip_path,                  # zip_download
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
        video_input = gr.Video(label="Upload Video (max 2:30)")
        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    # --- Outputs ---
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

    # Wire up the button
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, api_key_input],
        outputs=[
            video_1, video_2, video_3,
            video_4, video_5, video_6,
            heatmap_gallery, peak_gallery,
            report_output,
            html_download, pdf_download, zip_download,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
