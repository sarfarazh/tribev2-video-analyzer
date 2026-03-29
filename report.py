"""LLM report generation via OpenRouter and HTML export."""

import base64
import json
import logging
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

Structure your report as:

## Executive Summary
One paragraph: overall cognitive impact of this video. What does it primarily engage?

## Timeline Analysis
Walk through the video second-by-second (grouped by 20s segments). For each notable moment:
- What brain networks activate and what that means in plain language
- What likely caused it (scene cut, speech, music, new visual, etc.)
- Whether it represents strong engagement or a lull

## Peak Engagement Moments
The top 3-5 moments where brain activation is highest. Explain what makes each moment work — \
what the viewer's brain is doing and why that's good for engagement.

## Engagement Drops
Moments where activation falls. Explain what might be causing viewers to disengage and how to fix it.

## Brain Heatmap Interpretation
Reference the attached brain images. Point out key patterns visible in the heatmaps — which areas \
light up during the best moments vs. the weakest.

## Actionable Recommendations
5-7 specific, practical suggestions for future content based on the brain data. Be concrete \
(e.g., "Add a visual scene change around the 35-second mark to re-engage the visual cortex" \
not "Make your videos more engaging").

Rules:
- Use plain language throughout. If you must use a brain term, immediately explain it in parentheses.
- Always reference specific timestamps (e.g., "at 0:12").
- Focus on what's actionable for a content creator.
- Be honest about limitations — this is a predictive model, not a real brain scan.\
"""


def generate_report(
    api_key: str,
    summary: dict,
    heatmap_paths: list[str],
    peak_paths: list[str],
    model: str = "anthropic/claude-opus-4-6",
) -> str:
    """Generate a layman-friendly report using an LLM via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        summary: Full analysis summary dict (from analysis.build_full_summary).
        heatmap_paths: Paths to 5-second interval heatmap PNGs.
        peak_paths: Paths to peak/drop snapshot PNGs.
        model: OpenRouter model identifier.

    Returns:
        Report text in markdown format.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Build multimodal message content
    content = [
        {
            "type": "text",
            "text": (
                "Here is the brain activation data for my video:\n\n"
                f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
                "The following images show brain surface heatmaps at 5-second intervals "
                "and at peak/drop moments. Warmer colors (red/yellow) = stronger activation."
            ),
        },
    ]

    # Add heatmap images
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


def build_html_report(
    report_markdown: str,
    heatmap_paths: list[str],
    peak_paths: list[str],
) -> str:
    """Build a standalone HTML report with embedded images.

    Args:
        report_markdown: The LLM-generated report in markdown.
        heatmap_paths: Paths to heatmap PNGs to embed.
        peak_paths: Paths to peak/drop PNGs to embed.

    Returns:
        HTML string.
    """
    # Embed images as base64
    def img_tag(path: str, caption: str) -> str:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return (
                f'<figure style="display:inline-block;margin:8px;text-align:center">'
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px">'
                f'<figcaption style="font-size:0.85em;color:#666">{caption}</figcaption>'
                f'</figure>'
            )
        except FileNotFoundError:
            return ""

    heatmap_html = "".join(
        img_tag(p, f"Heatmap") for p in heatmap_paths
    )
    peak_html = "".join(
        img_tag(p, f"Peak/Drop") for p in peak_paths
    )

    # Convert markdown to HTML (basic conversion — avoid extra dependencies)
    # Gradio renders markdown natively, but for the download we do simple conversion
    report_html = report_markdown
    for level in range(6, 0, -1):
        prefix = "#" * level
        report_html = report_html.replace(
            f"\n{prefix} ", f"\n<h{level}>"
        ).replace(f"\n<h{level}>", f"\n<h{level}>")
    # Wrap paragraphs
    lines = report_html.split("\n")
    processed = []
    for line in lines:
        if line.startswith("<h"):
            # Close heading tag
            for level in range(1, 7):
                if line.startswith(f"<h{level}>"):
                    line = line + f"</h{level}>"
                    break
            processed.append(line)
        elif line.strip() == "":
            processed.append("")
        else:
            processed.append(f"<p>{line}</p>")
    report_html = "\n".join(processed)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRIBE Analyzer Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        line-height: 1.6;
        color: #1a1a1a;
        background: #fafafa;
    }}
    h1 {{ color: #2d3748; border-bottom: 3px solid #e53e3e; padding-bottom: 0.5rem; }}
    h2 {{ color: #4a5568; margin-top: 2rem; }}
    .images-section {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 1rem 0;
        justify-content: center;
    }}
    figure {{ background: white; padding: 8px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    p {{ margin: 0.5rem 0; }}
    @media print {{
        body {{ background: white; }}
        figure {{ box-shadow: none; border: 1px solid #ddd; }}
    }}
</style>
</head>
<body>
<h1>TRIBE Analyzer — Brain Response Report</h1>

<h2>Brain Activity Heatmaps (5-second intervals)</h2>
<div class="images-section">{heatmap_html}</div>

<h2>Peak &amp; Drop Moments</h2>
<div class="images-section">{peak_html}</div>

<hr>
{report_html}

<hr>
<p style="font-size:0.8em;color:#999;margin-top:2rem">
Generated by TRIBE Analyzer using Meta TRIBE v2 brain encoding model.
Brain predictions are model-based estimates, not real fMRI scans.
</p>
</body>
</html>"""
