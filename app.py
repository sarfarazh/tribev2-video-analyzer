"""TRIBE Analyzer — Gradio app with Script, Voiceover, Video, History, and Compare tabs."""

import os

# Must be set before any HuggingFace imports — RunPod sets this to 1
# but hf_transfer is not installed, causing ImportError during model download.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.5"

import json
import logging
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
    text_to_speech,
    trim_audio,
    trim_video,
)
from visuals import (
    generate_interval_heatmaps,
    generate_network_timeline,
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
    load_all_segment_derived,
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
    load_analysis_metadata,
)
from compare import (
    build_comparison_data,
    build_metrics_table,
    generate_comparison_report,
    generate_comparison_timeline,
    generate_delta_chart,
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
        files["timeline_path"],
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
        meta_path = analysis_dir / "metadata.json"
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

        if i in predict_done:
            cached = load_segment_predictions(analysis_dir, i)
        else:
            cached = None

        if cached:
            logger.info(f"Segment {i + 1}: loaded predictions from cache")
            result = cached
        else:
            logger.info(f"Segment {i + 1}: running TRIBE v2 on {seg['path']}")
            brain_type = "audio" if input_type == "script" else input_type
            result = process_segment(str(seg["path"]), input_type=brain_type)
            save_segment_predictions(
                analysis_dir, i,
                result["preds"], result["segments"], result["events"],
            )
            update_checkpoint(analysis_dir, "tribe_predict", segment_index=i)

        all_preds.append(result["preds"])
        all_segments.append(result["segments"])
        all_events.append(result["events"])

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
        frac = 0.50 + (i / n_segs) * 0.15
        progress(frac, desc=f"Rendering visualizations for segment {i + 1}...")

        hm_dir = analysis_dir / f"heatmaps_seg{i}"
        pk_dir = analysis_dir / f"peaks_seg{i}"
        dr_dir = analysis_dir / f"drops_seg{i}"
        mp4_out = analysis_dir / "videos" / f"segment_{i}.mp4"
        gif_out = analysis_dir / "gifs" / f"segment_{i}.gif"

        if i in vis_done and hm_dir.exists() and gif_out.exists():
            logger.info(f"Segment {i + 1}: loaded visuals from cache")
            import re
            for p in sorted(hm_dir.glob("*.png")):
                time_match = re.search(r"(\d+\.?\d*)s", p.stem)
                abs_time = float(time_match.group(1)) if time_match else 0
                time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"
                all_heatmaps.append({"path": str(p), "time_label": time_label})
            for d in [pk_dir, dr_dir]:
                if d.exists():
                    for p in sorted(d.glob("*.png")):
                        time_match = re.search(r"(\d+\.?\d*)s", p.stem)
                        abs_time = float(time_match.group(1)) if time_match else 0
                        time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"
                        all_peak_images.append({"path": str(p), "time_label": time_label})
            if mp4_out.exists():
                all_mp4s.append(str(mp4_out))
            all_gifs.append(str(gif_out))
            continue

        heatmaps = generate_interval_heatmaps(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"], interval=5,
            output_dir=hm_dir,
        )
        all_heatmaps.extend(heatmaps)

        peak_indices = [t for t, _ in all_peaks_drops[i]["peaks"]]
        drop_indices = [t for t, _ in all_peaks_drops[i]["drops"]]

        peaks = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=peak_indices, time_offset=seg["start"],
            output_dir=pk_dir, label_prefix="peak",
        )
        drops = generate_peak_snapshots(
            plotter, all_preds[i], all_segments[i],
            peak_timesteps=drop_indices, time_offset=seg["start"],
            output_dir=dr_dir, label_prefix="drop",
        )
        all_peak_images.extend(peaks + drops)

        if input_type in ("video", "audio", "script"):
            mp4_path = generate_segment_mp4(
                plotter, all_preds[i], all_segments[i],
                segment_video_path=str(seg["path"]),
                time_offset=seg["start"], output_path=mp4_out,
            )
            all_mp4s.append(mp4_path)

        gif_path = generate_segment_gif(
            plotter, all_preds[i], all_segments[i],
            time_offset=seg["start"], output_path=gif_out, fps=1,
        )
        all_gifs.append(gif_path)
        update_checkpoint(analysis_dir, "visualize", segment_index=i)

    update_checkpoint(analysis_dir, "visualize", completed=True)

    # --- Stage 2b: Network timeline chart ---
    progress(0.70, desc="Generating network timeline...")
    timeline_path = analysis_dir / "network_timeline.png"
    if not timeline_path.exists():
        generate_network_timeline(
            all_network_activations, all_peaks_drops,
            media_segments, total_duration,
            output_path=timeline_path,
        )
    timeline_path_str = str(timeline_path)

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
            timeline_path=timeline_path_str,
        )
        save_report_markdown(analysis_dir, report_markdown)

    update_checkpoint(analysis_dir, "llm_report", completed=True)

    # --- Stage 5: Build downloadable reports ---
    progress(0.90, desc="Preparing downloads...")

    heatmap_paths = [h["path"] for h in all_heatmaps]
    peak_paths = [p["path"] for p in all_peak_images]

    html_report = build_html_report(
        report_markdown, heatmap_paths, peak_paths,
        gif_paths=all_gifs, timeline_path=timeline_path_str,
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
        timeline_path=timeline_path_str,
    )

    update_checkpoint(analysis_dir, "package", completed=True)
    mark_complete(analysis_dir)

    meta_path = analysis_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    update_index(analysis_dir.name, meta)

    logger.info(f"Analysis complete! Saved to {analysis_dir}")

    heatmap_gallery = [(h["path"], h["time_label"]) for h in all_heatmaps]
    peak_gallery = [(p["path"], p["time_label"]) for p in all_peak_images]

    while len(all_mp4s) < MAX_SEGMENTS:
        all_mp4s.append(None)

    return (
        all_mp4s[0], all_mp4s[1], all_mp4s[2],
        all_mp4s[3], all_mp4s[4], all_mp4s[5],
        heatmap_gallery,
        peak_gallery,
        timeline_path_str,
        report_markdown,
        str(html_path),
        str(pdf_path) if pdf_path and pdf_path.exists() else None,
        zip_path,
    )


# ---------------------------------------------------------------------------
# Tab-specific entry points
# ---------------------------------------------------------------------------

def analyze_script(script_text: str, api_key: str, progress=gr.Progress()):
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if not script_text or not script_text.strip():
        raise gr.Error("Please enter your script text.")

    text = script_text.strip()
    input_hash = compute_input_hash(text)

    progress(0.02, desc="Converting script to speech (Chatterbox TTS)...")
    audio_path = text_to_speech(text)

    duration = get_media_duration(audio_path)
    if duration > 120:
        progress(0.05, desc="Trimming TTS audio to 120s...")
        audio_path = trim_audio(audio_path, max_duration=120)
        duration = 120

    progress(0.08, desc="Splitting TTS audio into segments...")
    audio_segments = split_audio(audio_path, segment_duration=20)

    label = text[:60] + ("..." if len(text) > 60 else "")

    return _run_pipeline(
        input_type="script", api_key=api_key, media_segments=audio_segments,
        total_duration=duration, input_hash=input_hash, input_label=label,
        progress=progress,
    )


def analyze_voiceover(audio_file: str, api_key: str, progress=gr.Progress()):
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
        audio_path = trim_audio(audio_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting audio into segments...")
    audio_segments = split_audio(audio_path, segment_duration=20)

    return _run_pipeline(
        input_type="audio", api_key=api_key, media_segments=audio_segments,
        total_duration=duration, input_hash=input_hash, input_label=audio_path.name,
        progress=progress,
    )


def analyze_video(video_file: str, api_key: str, progress=gr.Progress()):
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
        video_path = trim_video(video_path, max_duration=120)
        duration = 120

    progress(0.05, desc="Splitting video into segments...")
    video_segments = split_video(video_path, segment_duration=20)

    return _run_pipeline(
        input_type="video", api_key=api_key, media_segments=video_segments,
        total_duration=duration, input_hash=input_hash, input_label=video_path.name,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# History tab handlers
# ---------------------------------------------------------------------------

def refresh_history():
    index = load_index()
    if not index:
        return []
    rows = []
    for entry in reversed(index):
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
    if not analysis_id or not analysis_id.strip():
        raise gr.Error("Please select an analysis from the table first.")

    analysis_dir = RESULTS_DIR / analysis_id.strip()
    if not analysis_dir.exists():
        raise gr.Error(f"Analysis not found: {analysis_id}")

    files = list_analysis_files(analysis_dir)
    if not files["report_markdown"]:
        raise gr.Error("This analysis is incomplete (no report). Use Re-generate Report or re-run.")

    return _build_output_tuple(files)


def delete_history_entry(analysis_id: str):
    if not analysis_id or not analysis_id.strip():
        raise gr.Error("Please select an analysis first.")
    delete_analysis(analysis_id.strip())
    return refresh_history()


def regenerate_report(analysis_id: str, api_key: str, progress=gr.Progress()):
    """Re-generate only the LLM report using cached predictions."""
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenRouter API key.")
    if not analysis_id or not analysis_id.strip():
        raise gr.Error("Please enter an Analysis ID.")

    analysis_id = analysis_id.strip()
    analysis_dir = RESULTS_DIR / analysis_id
    if not analysis_dir.exists():
        raise gr.Error(f"Analysis not found: {analysis_id}")

    meta = load_analysis_metadata(analysis_id)
    if not meta:
        raise gr.Error("Could not load analysis metadata.")

    input_type = meta.get("input_type", "video")

    # Verify derived data exists (needed for report context)
    if not load_all_segment_derived(analysis_dir):
        raise gr.Error("This analysis has incomplete data. Please re-run the full analysis.")

    # Load JSON summary
    json_path = analysis_dir / "analysis.json"
    if json_path.exists():
        with open(json_path) as f:
            full_summary = json.load(f)
    else:
        raise gr.Error("Analysis JSON not found. Please re-run the full analysis.")

    # Collect image paths
    files = list_analysis_files(analysis_dir)
    heatmap_paths = [p for p, _ in files["heatmap_gallery"]]
    peak_paths = [p for p, _ in files["peak_gallery"]]
    timeline_path = files.get("timeline_path")

    progress(0.3, desc="Generating report with Claude Opus 4.6...")
    logger.info(f"Re-generating report for {analysis_id}...")

    report_markdown = generate_report(
        api_key=api_key.strip(),
        summary=full_summary,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
        input_type=input_type,
        timeline_path=timeline_path,
    )
    save_report_markdown(analysis_dir, report_markdown)

    # Rebuild HTML/PDF/ZIP
    progress(0.8, desc="Rebuilding downloads...")
    gif_dir = analysis_dir / "gifs"
    all_gifs = [str(p) for p in sorted(gif_dir.glob("*.gif"))] if gif_dir.exists() else []

    html_report = build_html_report(
        report_markdown, heatmap_paths, peak_paths,
        gif_paths=all_gifs, timeline_path=timeline_path,
    )
    html_path = analysis_dir / "report.html"
    html_path.write_text(html_report)

    pdf_path = analysis_dir / "report.pdf"
    try:
        build_pdf_report(str(html_path), str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        pdf_path = None

    mp4_dir = analysis_dir / "videos"
    all_mp4s = [str(p) for p in sorted(mp4_dir.glob("*.mp4"))] if mp4_dir.exists() else []

    zip_path = build_zip_package(
        output_path=analysis_dir / "tribe_analysis.zip",
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_path and pdf_path.exists() else None,
        json_path=str(json_path),
        mp4_paths=all_mp4s,
        gif_paths=all_gifs,
        heatmap_paths=heatmap_paths,
        peak_paths=peak_paths,
        timeline_path=timeline_path,
    )

    logger.info(f"Report re-generated for {analysis_id}")

    # Reload and return
    files = list_analysis_files(analysis_dir)
    return _build_output_tuple(files)


# ---------------------------------------------------------------------------
# Compare tab handler
# ---------------------------------------------------------------------------

def run_comparison(id_a: str, id_b: str, use_llm: bool, api_key: str, progress=gr.Progress()):
    if not id_a or not id_a.strip() or not id_b or not id_b.strip():
        raise gr.Error("Please enter both Analysis IDs.")

    id_a = id_a.strip()
    id_b = id_b.strip()

    progress(0.1, desc="Loading analysis data...")
    try:
        comparison_data = build_comparison_data(id_a, id_b)
    except ValueError as e:
        raise gr.Error(str(e))

    progress(0.3, desc="Generating comparison charts...")
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_compare_"))

    timeline_path = generate_comparison_timeline(
        comparison_data, output_path=tmp_dir / "comparison_timeline.png",
    )
    delta_path = generate_delta_chart(
        comparison_data, output_path=tmp_dir / "delta_chart.png",
    )

    progress(0.5, desc="Building metrics table...")
    metrics_rows = build_metrics_table(comparison_data)

    report_md = ""
    if use_llm:
        if not api_key or not api_key.strip():
            raise gr.Error("API key required for LLM comparison report.")
        progress(0.6, desc="Generating comparison report with Claude...")
        report_md = generate_comparison_report(
            api_key=api_key.strip(),
            comparison_data=comparison_data,
            timeline_path=timeline_path,
            delta_path=delta_path,
        )

    progress(1.0, desc="Done!")
    return timeline_path, delta_path, metrics_rows, report_md


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
            audio_input = gr.Audio(label="Upload Audio (max 2:30)", type="filepath")
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
                interactive=False, wrap=True,
            )
            selected_analysis_id = gr.Textbox(
                label="Analysis ID",
                placeholder="e.g. 20260330_141523_video_a1b2c3d4",
            )
            with gr.Row():
                load_btn = gr.Button("Load Analysis", variant="primary")
                regen_btn = gr.Button("Re-generate Report", variant="secondary")
                delete_btn = gr.Button("Delete Analysis", variant="stop")

        with gr.Tab("Compare"):
            gr.Markdown(
                "Select two past analyses to compare side by side. "
                "Use Analysis IDs from the History tab."
            )
            with gr.Row():
                compare_id_a = gr.Textbox(label="Analysis A (ID)", placeholder="Paste Analysis ID")
                compare_id_b = gr.Textbox(label="Analysis B (ID)", placeholder="Paste Analysis ID")
            compare_llm_toggle = gr.Checkbox(
                label="Generate LLM comparative report (uses API key)", value=False,
            )
            compare_btn = gr.Button("Compare", variant="primary", size="lg")

            gr.Markdown("### Comparison Results")
            compare_timeline_img = gr.Image(label="Network Timelines (A vs B)")
            compare_delta_img = gr.Image(label="Network Delta (A minus B)")
            gr.Markdown("### Key Metrics")
            compare_metrics_table = gr.Dataframe(
                headers=["Metric", "Analysis A", "Analysis B", "Delta"],
                interactive=False,
            )
            gr.Markdown("### Comparative Report")
            compare_report_md = gr.Markdown()

    # --- Shared outputs ---
    gr.Markdown("---")
    gr.Markdown("## Brain Activity Videos")
    with gr.Row():
        video_1 = gr.Video(label="Segment 1 (0:00-0:20)")
        video_2 = gr.Video(label="Segment 2 (0:20-0:40)")
        video_3 = gr.Video(label="Segment 3 (0:40-1:00)")
    with gr.Row():
        video_4 = gr.Video(label="Segment 4 (1:00-1:20)")
        video_5 = gr.Video(label="Segment 5 (1:20-1:40)")
        video_6 = gr.Video(label="Segment 6 (1:40-2:00)")

    gr.Markdown("## 5-Second Interval Heatmaps")
    heatmap_gallery = gr.Gallery(label="Brain activity every 5 seconds", columns=4, height="auto")

    gr.Markdown("## Peak & Drop Moments")
    peak_gallery = gr.Gallery(label="Highest activation peaks and lowest drops", columns=5, height="auto")

    gr.Markdown("## Network Timeline")
    timeline_image = gr.Image(label="7 Yeo Networks Over Time", type="filepath")

    gr.Markdown("## Analysis Report")
    report_output = gr.Markdown(label="Report")

    gr.Markdown("## Downloads")
    with gr.Row():
        html_download = gr.File(label="Report (HTML)")
        pdf_download = gr.File(label="Report (PDF)")
        zip_download = gr.File(label="Download All (ZIP)")

    # --- Output list (order must match _build_output_tuple) ---
    outputs = [
        video_1, video_2, video_3,
        video_4, video_5, video_6,
        heatmap_gallery, peak_gallery,
        timeline_image,
        report_output,
        html_download, pdf_download, zip_download,
    ]

    # Wire up analysis buttons
    script_btn.click(fn=analyze_script, inputs=[script_input, api_key_input], outputs=outputs)
    audio_btn.click(fn=analyze_voiceover, inputs=[audio_input, api_key_input], outputs=outputs)
    video_btn.click(fn=analyze_video, inputs=[video_input, api_key_input], outputs=outputs)

    # Wire up history tab
    history_refresh_btn.click(fn=refresh_history, outputs=history_table)
    load_btn.click(fn=load_history_entry, inputs=[selected_analysis_id], outputs=outputs)
    regen_btn.click(
        fn=regenerate_report,
        inputs=[selected_analysis_id, api_key_input],
        outputs=outputs,
    )
    delete_btn.click(fn=delete_history_entry, inputs=[selected_analysis_id], outputs=history_table)

    # Wire up compare tab
    compare_btn.click(
        fn=run_comparison,
        inputs=[compare_id_a, compare_id_b, compare_llm_toggle, api_key_input],
        outputs=[compare_timeline_img, compare_delta_img, compare_metrics_table, compare_report_md],
    )

    # Load history on page load
    demo.load(fn=refresh_history, outputs=history_table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
