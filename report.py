"""LLM report generation via OpenRouter, HTML/PDF export, and ZIP packaging."""

import base64
import json
import logging
import re
import shutil
import zipfile
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert neuroscience communicator helping a content creator (Instagram Reels / YouTube Shorts) \
understand how their video affects the brain. You will receive:

1. A JSON summary of brain network activations over time (per-second, grouped into 20-second segments)
2. Brain heatmap images showing cortical activity at 5-second intervals and at peak/drop moments

The 7 brain networks measured are:
- Visual: processing what viewers see (scene changes, faces, motion, color)
- Somatomotor: physical sensation and movement response
- Dorsal Attention: focused, voluntary attention (tracking objects, following action)
- Ventral Attention: surprise, alertness, detecting unexpected events
- Limbic: emotional response, reward, motivation
- Frontoparietal: thinking, decision-making, working memory
- Default Mode: personal connection, storytelling, empathy, self-referential thought

Write your report for a content creator who has NO neuroscience background. Your goal is to help them \
understand what their video does to viewers' brains and how to make better content.

IMPORTANT FORMATTING RULES:
- Use proper markdown: **bold** for emphasis, bullet lists with `-`, numbered lists with `1.`
- Keep it scannable — use short paragraphs, not walls of text
- No emojis in headings
- Always reference specific timestamps (e.g., "at 0:12")
- Use plain language. If you must use a brain term, explain it in parentheses.
- Be honest about limitations — this is a predictive model, not a real brain scan.

Structure your report EXACTLY as follows:

## Executive Summary

One short paragraph (3-4 sentences max): overall cognitive impact of this video. What does it primarily engage?

## Timeline Overview

Walk through the video grouped by 20s segments. For each segment, write a detailed paragraph covering:
- The overall engagement level and arc within the segment
- Key moments with specific timestamps and what brain networks they triggered
- What content elements (narration, visuals, music, silence) caused the activations or drops
- How the segment connects to the next

Reference specific timestamps (e.g., "at 0:12") and activation values. \
Explain what each brain network response means in plain language.

## Top 3 Peak Moments

List EXACTLY 3 peak moments, numbered 1-3, sorted in CHRONOLOGICAL order (earliest first). Each as:

**1. [Timestamp] — [Short title]** (Activation: X.XX)

[2-3 sentences: what the brain does here, why it works for engagement, what content element caused it]

## Top 3 Engagement Drops

List EXACTLY 3 drop moments, numbered 1-3, sorted in CHRONOLOGICAL order (earliest first). Each as:

**1. [Timestamp] — [Short title]** (Activation: X.XX)

[2-3 sentences: what's happening in the brain, what likely caused the drop, how to fix it]

## What to Change (Edit & Retest)

The goal is to maximize cognitive engagement so the video has the best chance of going viral. \
Viral content keeps brains firing on multiple networks simultaneously — especially attention, emotion, and default mode (personal connection).

Structure this section as TWO parts:

### Fix the Weak Spots

For each engagement drop identified above, give ONE specific edit the creator can make to fix it. \
Format as a numbered list matching the drops:

1. **[Timestamp]**: [Specific edit — e.g., "Replace the static shot with a quick zoom-in or scene cut to re-trigger visual attention" or "Add a surprising sound effect here to spike the alertness network"]

### Double Down on What Works

For each peak moment, explain how to amplify or replicate that pattern elsewhere in the video:

1. **[Timestamp]**: [Specific suggestion — e.g., "The emotional payoff here is strong. Build more tension in the 5 seconds before it to make the peak even higher" or "This hook pattern works — reuse it around 0:35 where engagement dips"]

### General Virality Tips (Based on This Video's Brain Data)

3-5 bullet points of broader patterns the creator should apply to future content. \
Tie each tip to specific brain data from this video. Be concrete and actionable — \
the creator will edit the video based on these suggestions and re-upload it to test again.
"""


INPUT_TYPE_CONTEXT = {
    "video": (
        "Here is the brain activation data for my video:\n\n"
    ),
    "audio": (
        "This analysis is based on audio only (a voiceover recording). "
        "Visual network data reflects imagined/default neural responses to auditory stimuli, "
        "not actual visual content. Focus your analysis on how the voice, tone, pacing, "
        "music, and sound design affect the brain.\n\n"
        "Here is the brain activation data for my voiceover:\n\n"
    ),
    "script": (
        "This analysis is based on a text script that was converted to speech using "
        "Chatterbox TTS. The brain responses are predicted from the synthesized audio. "
        "Focus your analysis on how the script's narrative structure, word choice, pacing, "
        "and emotional arc affect the brain.\n\n"
        "Here is the brain activation data for my script:\n\n"
    ),
}


def generate_report(
    api_key: str,
    summary: dict,
    heatmap_paths: list[str],
    peak_paths: list[str],
    model: str = "anthropic/claude-opus-4-6",
    input_type: str = "video",
    timeline_path: str | None = None,
) -> str:
    """Generate a layman-friendly report using an LLM via OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    context_prefix = INPUT_TYPE_CONTEXT.get(input_type, INPUT_TYPE_CONTEXT["video"])

    content = [
        {
            "type": "text",
            "text": (
                context_prefix
                + f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
                "The following images show brain surface heatmaps at 5-second intervals "
                "and at peak/drop moments. Warmer colors (red/yellow) = stronger activation."
            ),
        },
    ]

    # Include timeline chart if available
    if timeline_path:
        try:
            with open(timeline_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "text",
                "text": "This line chart shows all 7 brain networks plotted over time. "
                        "Peaks are marked with red triangles, drops with blue triangles.",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except FileNotFoundError:
            pass

    all_images = [
        (p, "5-second interval heatmap") for p in heatmap_paths
    ] + [
        (p, "peak/drop moment") for p in peak_paths
    ]

    for img_path, label in all_images:
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except FileNotFoundError:
            logger.warning(f"Image not found, skipping: {img_path}")

    logger.info(
        f"Sending {len(all_images)} images + JSON to {model} via OpenRouter..."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def _md_to_html(md: str) -> str:
    """Convert markdown to HTML without external dependencies."""
    html = md

    for level in range(6, 0, -1):
        pattern = re.compile(r"^" + "#" * level + r" (.+)$", re.MULTILINE)
        html = pattern.sub(rf"<h{level}>\1</h{level}>", html)

    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    def _replace_ul(match):
        items = re.findall(r"^- (.+)$", match.group(0), re.MULTILINE)
        li = "".join(f"<li>{item}</li>\n" for item in items)
        return f"<ul>\n{li}</ul>"
    html = re.sub(r"(^- .+$\n?)+", _replace_ul, html, flags=re.MULTILINE)

    def _replace_ol(match):
        items = re.findall(r"^\d+\. (.+)$", match.group(0), re.MULTILINE)
        li = "".join(f"<li>{item}</li>\n" for item in items)
        return f"<ol>\n{li}</ol>"
    html = re.sub(r"(^\d+\. .+$\n?)+", _replace_ol, html, flags=re.MULTILINE)

    html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

    lines = html.split("\n")
    processed = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            processed.append("")
        elif stripped.startswith("<"):
            processed.append(line)
        else:
            processed.append(f"<p>{line}</p>")
    html = "\n".join(processed)

    return html


def _embed_file_b64(path: str, mime: str) -> str:
    """Read a file and return a base64 data URI."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except FileNotFoundError:
        return ""


def build_html_report(
    report_markdown: str,
    heatmap_paths: list[str],
    peak_paths: list[str],
    gif_paths: list[str] | None = None,
    timeline_path: str | None = None,
) -> str:
    """Build a standalone HTML report with embedded images and GIFs."""

    def img_tag(path: str, caption: str) -> str:
        uri = _embed_file_b64(path, "image/png")
        if not uri:
            return ""
        return (
            f'<figure>'
            f'<img src="{uri}">'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )

    def gif_tag(path: str, caption: str) -> str:
        uri = _embed_file_b64(path, "image/gif")
        if not uri:
            return ""
        return (
            f'<figure class="gif-figure">'
            f'<img src="{uri}">'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )

    # Separate peak images from drop images by filename
    peak_imgs = []
    drop_imgs = []
    for p in peak_paths:
        fname = Path(p).name
        if "drop" in fname:
            drop_imgs.append(p)
        else:
            peak_imgs.append(p)

    # Build heatmap gallery with timestamps
    heatmap_html = ""
    for p in heatmap_paths:
        fname = Path(p).stem
        time_match = re.search(r"(\d+\.?\d*)s", fname)
        if time_match:
            secs = float(time_match.group(1))
            label = f"{int(secs // 60)}:{int(secs % 60):02d}"
        else:
            label = "Heatmap"
        heatmap_html += img_tag(p, label)

    peak_html = "".join(
        img_tag(p, f"Peak {i + 1}") for i, p in enumerate(peak_imgs)
    )
    drop_html = "".join(
        img_tag(p, f"Drop {i + 1}") for i, p in enumerate(drop_imgs)
    )

    # Build GIF section
    gif_html = ""
    if gif_paths:
        for i, gp in enumerate(gif_paths):
            if gp:
                start = i * 20
                end = start + 20
                label = f"Segment {i + 1} ({int(start // 60)}:{start % 60:02d}–{int(end // 60)}:{end % 60:02d})"
                gif_html += gif_tag(gp, label)

    # Timeline chart
    timeline_html = ""
    if timeline_path:
        uri = _embed_file_b64(timeline_path, "image/png")
        if uri:
            timeline_html = (
                f'<figure style="max-width:100%;margin:1rem auto;">'
                f'<img src="{uri}" style="width:100%;border-radius:8px;">'
                f'<figcaption>All 7 brain networks over time. Red triangles = peaks, blue triangles = drops.</figcaption>'
                f'</figure>'
            )

    report_html = _md_to_html(report_markdown)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRIBE Analyzer Report</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 960px;
        margin: 0 auto;
        padding: 2rem;
        line-height: 1.7;
        color: #1a1a1a;
        background: #fafafa;
    }}
    h1 {{
        color: #2d3748;
        border-bottom: 3px solid #e53e3e;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    h2 {{
        color: #2d3748;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e2e8f0;
    }}
    h3 {{
        color: #4a5568;
        margin-top: 1.5rem;
    }}
    p {{ margin: 0.6rem 0; }}
    strong {{ color: #2d3748; }}
    ul, ol {{
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }}
    li {{ margin: 0.3rem 0; }}
    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 2rem 0;
    }}
    .images-section {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 1rem 0;
        justify-content: center;
    }}
    .gif-section {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 1rem 0;
        justify-content: center;
    }}
    figure {{
        display: inline-block;
        margin: 8px;
        text-align: center;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .gif-figure {{
        max-width: 320px;
    }}
    figure img {{
        max-width: 100%;
        border-radius: 8px;
    }}
    figcaption {{
        font-size: 0.85em;
        color: #666;
        margin-top: 4px;
        font-weight: 500;
    }}
    .report-body h2 {{ border-left: 4px solid #4299e1; padding-left: 0.75rem; border-bottom: none; }}
    .footer {{
        font-size: 0.8em;
        color: #999;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }}
    @media print {{
        body {{ background: white; padding: 1rem; }}
        figure {{ box-shadow: none; border: 1px solid #ddd; }}
        .gif-figure img {{ /* Show first frame only in print */ }}
    }}
</style>
</head>
<body>

<h1>TRIBE Analyzer — Brain Response Report</h1>

{"<h2>Brain Activity Animations</h2>" + '<div class="gif-section">' + gif_html + "</div>" if gif_html else ""}

{"<h2>Network Timeline</h2>" + timeline_html if timeline_html else ""}

<h2>Brain Activity Heatmaps</h2>
<p style="color:#666;font-size:0.9em">Cortical surface plots at 5-second intervals. Warmer colors = stronger activation.</p>
<div class="images-section">{heatmap_html}</div>

<h2>Peak Engagement Moments</h2>
<div class="images-section">{peak_html}</div>

<h2>Engagement Drops</h2>
<div class="images-section">{drop_html}</div>

<hr>

<div class="report-body">
{report_html}
</div>

<div class="footer">
Generated by TRIBE Analyzer using Meta TRIBE v2 brain encoding model.
Brain predictions are model-based estimates, not real fMRI scans.
</div>

</body>
</html>"""


def build_pdf_report(html_path: str | Path, output_path: str | Path) -> str:
    """Convert an HTML report to a multi-page PDF using weasyprint.

    Args:
        html_path: Path to the HTML report file.
        output_path: Where to save the PDF.

    Returns:
        Path to the generated PDF file.
    """
    from weasyprint import HTML

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    HTML(filename=str(html_path)).write_pdf(str(output_path))
    logger.info(f"PDF report saved to {output_path}")

    return str(output_path)


def build_zip_package(
    output_path: str | Path,
    html_path: str | None = None,
    pdf_path: str | None = None,
    json_path: str | None = None,
    mp4_paths: list[str] | None = None,
    gif_paths: list[str] | None = None,
    heatmap_paths: list[str] | None = None,
    peak_paths: list[str] | None = None,
    timeline_path: str | None = None,
) -> str:
    """Build an organized ZIP file with all analysis outputs.

    Args:
        output_path: Where to save the ZIP file.
        html_path: Path to HTML report.
        pdf_path: Path to PDF report.
        json_path: Path to analysis JSON.
        mp4_paths: Paths to segment MP4 videos.
        gif_paths: Paths to segment GIFs.
        heatmap_paths: Paths to heatmap PNGs.
        peak_paths: Paths to peak/drop PNGs.

    Returns:
        Path to the generated ZIP file.
    """
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if html_path:
            zf.write(html_path, "tribe_analysis/report.html")
        if pdf_path:
            zf.write(pdf_path, "tribe_analysis/report.pdf")
        if json_path:
            zf.write(json_path, "tribe_analysis/data/analysis.json")

        if mp4_paths:
            for i, p in enumerate(mp4_paths):
                if p:
                    zf.write(p, f"tribe_analysis/videos/segment_{i + 1}.mp4")

        if gif_paths:
            for i, p in enumerate(gif_paths):
                if p:
                    zf.write(p, f"tribe_analysis/gifs/segment_{i + 1}.gif")

        if heatmap_paths:
            for p in heatmap_paths:
                fname = Path(p).name
                zf.write(p, f"tribe_analysis/heatmaps/{fname}")

        if peak_paths:
            for p in peak_paths:
                fname = Path(p).name
                zf.write(p, f"tribe_analysis/peaks/{fname}")

        if timeline_path:
            zf.write(timeline_path, "tribe_analysis/network_timeline.png")

    logger.info(f"ZIP package saved to {output_path}")
    return str(output_path)
