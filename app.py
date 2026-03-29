"""TRIBE Analyzer — Gradio app with Script, Voiceover, Video, and History tabs."""

import os

# Must be set before any HuggingFace imports — RunPod sets this to 1
# but hf_transfer is not installed, causing ImportError during model download.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.5"

import logging
import shutil
import subprocess
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
from persistence import (
    RESULTS_DIR,
    compute_input_hash,
    delete_analysis,
    find_existing_analysis,
    generate_analysis_id,
    init_analysis,
    list_analysis_files,
    load_checkpoint,
    load_index,
    load_report_markdown,
    load_segment_derived,
    load_segment_predictions,
    mark_complete,
    save_report_markdown,
    save_segment_derived,
    save_segment_predictions,
    update_checkpoint,
    update_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MAX_SEGMENTS = 6  # Max video/audio slots in UI (120s / 20s)

# Load model at startup
load_model(cache_folder=str(CACHE_FOLDER))


# ---------------------------------------------------------------------------
# Helper: build output tuple from files dict
# ---------------------------------------------------------------------------

def _build_output_tuple(files: dict) -> tuple:
    """Convert a files dict (from list_analysis_files) to the Gradio output tuple."""
    mp4s = files["mp4_paths"]
    while len(mp4s) < MAX_SEGMENTS:
        mp4s.append(None)

    return (
        mp4s[0], mp4s[1], mp4s[2],
        mp4s[3], mp4s[4], mp4s[5],
        files["heatmap_gallery"],
        files["peak_gallery"],
        files["report_markdown"],
        files["html_path"],
        files["pdf_path"],
        files["zip_path"],
    )


# ---------------------------------------------------------------------------
# Shared analysis pipeline with checkpointing
# ---------------------------------------------------------------------------

def _run_pipeline(
    input_type: str,
    api_key: str,
    media_segments: list[dict],
    total_duration: float,
    input_hash: str,
    input_label: str,
    progress: gr.Progress,
) -> tuple:
    """Core pipeline shared by all three tabs, with checkpoint/resume support."""

    # --- Check for existing complete analysis ---
    existing = find_existing_analysis(input_type, input_hash)
    if existing:
        cp = load_checkpoint(existing)
        if cp.get("status") == "complete":
            logger.info(f"Found complete cached analysis: {existing.name}")
            progress(1.0, desc="Loaded from cache!")
            files = list_analysis_files(existing)
            if files["report_markdown"]:
                return _build_output_tuple(files)
            # If report is missing, fall through to re-generate
            logger.info("Cached analysis missing report — will resume")

    # --- Init or resume analysis ---
    if existing and cp.get("status") != "complete":
        analysis_dir = existing
        logger.info(f"Resuming partial analysis: {existing.name}")
    else:
        analysis_id = generate_analysis_id(input_type, input_hash)
        analysis_dir = init_analysis(
            analysis_id=analysis_id,
            input_type=input_type,
            duration=total_duration,
            label=input_label,
            input_hash=input_hash,
            n_segments=len(media_segments),
        )
        # Add to index
        meta_path = analysis_dir / "metadata.json"
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        update_index(analysis_dir.name, meta)
        logger.info(f"New analysis: {analysis_dir.name}")

    cp = load_checkpoint(analysis_dir)
    plotter = get_plotter()
    atlas_plotter = get_atlas_plotter()
    n_segs = len(media_segments)

    # --- Stage 1: TRIBE v2 predictions (per-segment checkpoint) ---
    all_preds = []
    all_segments = []
    all_events = []
    all_network_activations = []
    all_peaks_drops = []

    predict_done = set(cp.get("stages", {}).get("tribe_predict", {}).get("segments_done", []))
    derive_done = set(cp.get("stages", {}).get("derive", {}).get("segments_done", []))

    for i, seg in enumerate(media_segments):
        frac = 0.05 + (i / n_segs) * 0.40
        progress(frac, desc=f"Processing segment {i + 1}/{n_segs} with TRIBE v2...")

        # Try loading cached predictions
        if i in predict_done:
            cached = load_segment_predictions(analysis_dir, i)
        else:
            cached = None

        if cached:
            logger.info(f"Segment {i + 1}: loaded predictions from cache")
            result = cached
        else:
            logger.info(f"Segment {i + 1}: running TRIBE v2 on {seg['path']}")
            result = process_segment(str(seg["path"]), input_type=input_type)
            save_segment_predictions(
                analysis_dir, i,
                result["preds"], result["segments"], result["events"],
            )
            update_checkpoint(analysis_dir, "tribe_predict", segment_index=i)

        all_preds.append(result["preds"])
        all_segments.append(result["segments"])
        all_events.append(result["events"])

        # Derived data (network activations, peaks/drops)
        if i in derive_done:
            derived = load_segment_derived(analysis_dir, i)
        else:
            derived = None

        if derived:
            logger.info(f"Segment {i + 1}: loaded derived data from cache")
            net_act = derived["network_activations"]
            pd_result = derived["peaks_drops"]
        else:
            net_act = aggregate_to_networks(result["preds"], atlas_plotter)
            pd_result = find_peaks_and_drops(net_act)
            save_segment_derived(analysis_dir, i, net_act, pd_result)
            update_checkpoint(analysis_dir, "derive", segment_index=i)

        all_network_activations.append(net_act)
        all_peaks_drops.append(pd_result)

    update_checkpoint(analysis_dir, "tribe_predict", completed=True)
    update_checkpoint(analysis_dir, "derive", completed=True)

    # --- Stage 2: Generate visualizations ---
    progress(0.50, desc="Generating brain heatmaps...")
    logger.info("Generating visualizations...")

    all_heatmaps = []
    all_peak_images = []
    all_mp4s = []
    all_gifs = []

    vis_done = set(cp.get("stages", {}).get("visualize", {}).get("segments_done", []))

    for i, seg in enumerate(media_segments):
        frac = 0.50 + (i / n_segs) * 0.20
        progress(frac, desc=f"Rendering visualizations for segment {i + 1}...")

        hm_dir = analysis_dir / f"heatmaps_seg{i}"
        pk_dir = analysis_dir / f"peaks_seg{i}"
        dr_dir = analysis_dir / f"drops_seg{i}"
        mp4_out = analysis_dir / "videos" / f"segment_{i}.mp4"
        gif_out = analysis_dir / "gifs" / f"segment_{i}.gif"

        # Check if visuals already exist for this segment
        if i in vis_done and hm_dir.exists() and gif_out.exists():
            logger.info(f"Segment {i + 1}: loaded visuals from cache")
            # Rebuild heatmap metadata from files
            import re
            for p in sorted(hm_dir.glob("*.png")):
                time_match = re.search(r"(\d+\.?\d*)s", p.stem)
                abs_time = float(time_match.group(1)) if time_match else 0
                time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"
                all_heatmaps.append({
                    "path": str(p), "time_label": time_label,
                })
            for d in [pk_dir, dr_dir]:
                if d.exists():
                    for p in sorted(d.glob("*.png")):
                        time_match = re.search(r"(\d+\.?\d*)s", p.stem)
                        abs_time = float(time_match.group(1)) if time_match else 0
                        time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"
                        all_peak_images.append({
                            "path": str(p), "time_label": time_label,
                        })
            if mp4_out.exists():
                all_mp4s.append(str(mp4_out))
            all_gifs.append(str(gif_out))
            continue

        # Generate fresh
        heatmaps = generate_interval_heatmaps(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"],
            interval=5,
            output_dir=hm_dir,
        )
        all_heatmaps.extend(heatmaps)

        peak_indices = [t for t, _ in all_peaks_drops[i]["peaks"]]
        drop_indices = [t for t, _ in all_peaks_drops[i]["drops"]]

        peaks = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=peak_indices,
            time_offset=seg["start"],
            output_dir=pk_dir,
            label_prefix="peak",
        )
        drops = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=drop_indices,
            time_offset=seg["start"],
            output_dir=dr_dir,
            label_prefix="drop",
        )
        all_peak_images.extend(peaks + drops)

        # MP4 with audio (video/audio input types)
        if input_type in ("video", "audio"):
            mp4_path = generate_segment_mp4(
                plotter, all_preds[i], all_segments[i],
                segment_video_path=str(seg["path"]),
                time_offset=seg["start"],
                output_path=mp4_out,
            )
            all_mp4s.append(mp4_path)

        # GIF
        gif_path = generate_segment_gif(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"],
            output_path=gif_out,
            fps=1,
        )
        all_gifs.append(gif_path)

        update_checkpoint(analysis_dir, "visualize", segment_index=i)

    # For text input, convert GIFs to MP4 (no audio to mux)
    if input_type == "text":
        for i in range(len(media_segments)):
            gif_p = all_gifs[i] if i < len(all_gifs) else None
            mp4_out = analysis_dir / "videos" / f"segment_{i}.mp4"
            if gif_p and not mp4_out.exists():
                subprocess.run(
                    [
                        "ffmpeg", "-i", str(gif_p),
                        "-movflags", "faststart",
                        "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-y", str(mp4_out),
                    ],
                    capture_output=True,
                )
            if mp4_out.exists():
                all_mp4s.append(str(mp4_out))

    update_checkpoint(analysis_dir, "visualize", completed=True)

    # --- Stage 3: Build JSON summary ---
    progress(0.72, desc="Building analysis summary...")
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
    json_path = save_summary(full_summary, analysis_dir / "analysis.json")
    update_checkpoint(analysis_dir, "json_summary", completed=True)

    # --- Stage 4: LLM report (check cache first — saves tokens) ---
    cached_report = load_report_markdown(analysis_dir)
    if cached_report:
        logger.info("Loaded LLM report from cache (saved tokens!)")
        report_markdown = cached_report
    else:
        progress(0.78, desc="Generating report with Claude Opus 4.6...")
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
        # Save immediately — this is the expensive one
        save_report_markdown(analysis_dir, report_markdown)

    update_checkpoint(analysis_dir, "llm_report", completed=True)

    # --- Stage 5: Build downloadable reports ---
    progress(0.90, desc="Preparing downloads...")

    heatmap_paths = [h["path"] for h in all_heatmaps]
    peak_paths = [p["path"] for p in all_peak_images]

    html_report = build_html_report(
        report_markdown, heatmap_paths, peak_paths, gif_paths=all_gifs,
    )
    html_path = analysis_dir / "report.html"
    html_path.write_text(html_report)

    pdf_path = analysis_dir / "report.pdf"
    try:
        build_pdf_report(str(html_path), str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        pdf_path = None

    progress(0.95, desc="Building download package...")
    zip_path = build_zip_package(
        output_path=analysis_dir / "tribe_analysis.zip",
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_path and pdf_path.exists() else None,
        json_path=json_path,
        mp4_paths=all_mp4s,
        gif_paths=all_gifs,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
    )

    update_checkpoint(analysis_dir, "package", completed=True)

    # --- Mark complete ---
    mark_complete(analysis_dir)
    # Update index with final metadata
    import json
    meta_path = analysis_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    update_index(analysis_dir.name, meta)

    logger.info(f"Analysis complete! Saved to {analysis_dir}")

    # Build gallery data
    heatmap_gallery = [(h["path"], h["time_label"]) for h in all_heatmaps]
    peak_gallery = [(p["path"], p["time_label"]) for p in all_peak_images]

    while len(all_mp4s) < MAX_SEGMENTS:
        all_mp4s.append(None)

    return (
        all_mp4s[0], all_mp4s[1], all_mp4s[2],
        all_mp4s[3], all_mp4s[4], all_mp4s[5],
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

    text = script_text.strip()
    input_hash = compute_input_hash(text)

    progress(0.02, desc="Saving script to file...")
    text_path = text_to_file(text)

    segments = [{
        "path": text_path,
        "start": 0,
        "duration": 0,
        "index": 0,
    }]

    label = text[:60] + ("..." if len(text) > 60 else "")

    return _run_pipeline(
        input_type="text",
        api_key=api_key,
        media_segments=segments,
        total_duration=0,
        input_hash=input_hash,
        input_label=label,
        progress=progress,
    )


def analyze_voiceover(audio_file: str, api_key: str, progress=gr.Progress()):
    """Voiceover tab — audio file split and analyzed."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")

    audio_path = Path(audio_file)
    with open(audio_path, "rb") as f:
        input_hash = compute_input_hash(f.read())

    duration = get_media_duration(audio_path)

    if duration > 150:
        raise gr.Error(f"Audio is {duration:.0f}s — max supported is 2:30.")

    if duration > 120:
        progress(0.02, desc="Trimming audio to 120s...")
        logger.info(f"Audio is {duration:.0f}s — trimming to first 120s")
        audio_path = trim_audio(audio_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting audio into segments...")
    audio_segments = split_audio(audio_path, segment_duration=20)
    logger.info(f"Split into {len(audio_segments)} segments")

    return _run_pipeline(
        input_type="audio",
        api_key=api_key,
        media_segments=audio_segments,
        total_duration=duration,
        input_hash=input_hash,
        input_label=audio_path.name,
        progress=progress,
    )


def analyze_video(video_file: str, api_key: str, progress=gr.Progress()):
    """Video tab — video file split and analyzed."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    video_path = Path(video_file)
    with open(video_path, "rb") as f:
        input_hash = compute_input_hash(f.read())

    duration = get_media_duration(video_path)

    if duration > 150:
        raise gr.Error(f"Video is {duration:.0f}s — max supported is 2:30.")

    if duration > 120:
        progress(0.02, desc="Trimming video to 120s...")
        logger.info(f"Video is {duration:.0f}s — trimming to first 120s")
        video_path = trim_video(video_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting video into segments...")
    video_segments = split_video(video_path, segment_duration=20)
    logger.info(f"Split into {len(video_segments)} segments")

    return _run_pipeline(
        input_type="video",
        api_key=api_key,
        media_segments=video_segments,
        total_duration=duration,
        input_hash=input_hash,
        input_label=video_path.name,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# History tab handlers
# ---------------------------------------------------------------------------

def refresh_history():
    """Return history data for the Gradio Dataframe."""
    index = load_index()
    if not index:
        return []
    rows = []
    for entry in reversed(index):  # newest first
        created = entry.get("created_at", "")[:16].replace("T", " ")
        rows.append([
            entry.get("analysis_id", ""),
            created,
            entry.get("input_type", ""),
            f"{entry.get('duration_seconds', 0):.0f}s",
            entry.get("status", "unknown"),
            entry.get("label", "")[:40],
        ])
    return rows


def load_history_entry(analysis_id: str):
    """Load a past analysis and return the same output tuple as _run_pipeline."""
    if not analysis_id or not analysis_id.strip():
        raise gr.Error("Please select an analysis from the table first.")

    analysis_id = analysis_id.strip()
    analysis_dir = RESULTS_DIR / analysis_id

    if not analysis_dir.exists():
        raise gr.Error(f"Analysis not found: {analysis_id}")

    files = list_analysis_files(analysis_dir)
    if not files["report_markdown"]:
        raise gr.Error(
            "This analysis is incomplete (no report). "
            "Re-run the analysis to finish it."
        )

    return _build_output_tuple(files)


def delete_history_entry(analysis_id: str):
    """Delete an analysis and return refreshed history."""
    if not analysis_id or not analysis_id.strip():
        raise gr.Error("Please select an analysis first.")
    delete_analysis(analysis_id.strip())
    return refresh_history()


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

        with gr.Tab("History"):
            gr.Markdown(
                "View and reload past analyses. Results are stored locally — "
                "download ZIPs as backup before terminating the pod."
            )
            history_refresh_btn = gr.Button("Refresh", size="sm")
            history_table = gr.Dataframe(
                headers=["ID", "Date", "Type", "Duration", "Status", "Label"],
                interactive=False,
                wrap=True,
            )
            selected_analysis_id = gr.Textbox(
                label="Selected Analysis ID (click a row above, then paste the ID)",
                placeholder="e.g. 20260330_141523_video_a1b2c3d4",
            )
            with gr.Row():
                load_btn = gr.Button("Load Analysis", variant="primary")
                delete_btn = gr.Button("Delete Analysis", variant="stop")

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

    # --- Output list ---
    outputs = [
        video_1, video_2, video_3,
        video_4, video_5, video_6,
        heatmap_gallery, peak_gallery,
        report_output,
        html_download, pdf_download, zip_download,
    ]

    # Wire up analysis buttons
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

    # Wire up history tab
    history_refresh_btn.click(
        fn=refresh_history,
        outputs=history_table,
    )
    load_btn.click(
        fn=load_history_entry,
        inputs=[selected_analysis_id],
        outputs=outputs,
    )
    delete_btn.click(
        fn=delete_history_entry,
        inputs=[selected_analysis_id],
        outputs=history_table,
    )

    # Load history on page load
    demo.load(fn=refresh_history, outputs=history_table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
