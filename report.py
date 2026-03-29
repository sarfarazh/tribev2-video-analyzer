"""LLM report generation via OpenRouter and HTML export."""

import base64
import json
import logging
import re
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

Give a CONCISE overview of the video's brain engagement arc, grouped by 20s segments. \
For each segment, write a SHORT paragraph (3-5 sentences) covering:
- The overall engagement level (high/medium/low)
- The 2-3 most notable moments and what they triggered
- Any significant shifts in brain activity

DO NOT go second-by-second. Focus on the narrative arc and key transitions. \
Use a brief timestamp range like "0:12–0:15" to anchor observations.

## Top 3 Peak Moments

List EXACTLY 3 peak moments, numbered 1-3, each as:

**1. [Timestamp] — [Short title]** (Activation: X.XX)

[2-3 sentences: what the brain does here, why it works for engagement, what content element caused it]

## Top 3 Engagement Drops

List EXACTLY 3 drop moments, numbered 1-3, each as:

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


def _md_to_html(md: str) -> str:
    """Convert markdown to HTML without external dependencies."""
    html = md

    # Headings (process longest prefix first)
    for level in range(6, 0, -1):
        pattern = re.compile(r"^" + "#" * level + r" (.+)$", re.MULTILINE)
        html = pattern.sub(rf"<h{level}>\1</h{level}>", html)

    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

    # Italic
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Unordered lists: group consecutive lines starting with "- "
    def _replace_ul(match):
        items = re.findall(r"^- (.+)$", match.group(0), re.MULTILINE)
        li = "".join(f"<li>{item}</li>\n" for item in items)
        return f"<ul>\n{li}</ul>"
    html = re.sub(r"(^- .+$\n?)+", _replace_ul, html, flags=re.MULTILINE)

    # Ordered lists: group consecutive lines starting with "N. "
    def _replace_ol(match):
        items = re.findall(r"^\d+\. (.+)$", match.group(0), re.MULTILINE)
        li = "".join(f"<li>{item}</li>\n" for item in items)
        return f"<ol>\n{li}</ol>"
    html = re.sub(r"(^\d+\. .+$\n?)+", _replace_ol, html, flags=re.MULTILINE)

    # Horizontal rules
    html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

    # Wrap remaining plain text lines as paragraphs
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
    def img_tag(path: str, caption: str) -> str:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return (
                f'<figure>'
                f'<img src="data:image/png;base64,{b64}">'
                f'<figcaption>{caption}</figcaption>'
                f'</figure>'
            )
        except FileNotFoundError:
            return ""

    # Separate peak images from drop images by filename
    peak_imgs = []
    drop_imgs = []
    for p in peak_paths:
        fname = Path(p).name
        if "drop" in fname:
            drop_imgs.append(p)
        else:
            peak_imgs.append(p)

    # Build heatmap gallery with timestamps from filenames
    heatmap_html = ""
    for p in heatmap_paths:
        fname = Path(p).stem  # e.g. "heatmap_005.0s"
        # Extract time from filename
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
    figure {{
        display: inline-block;
        margin: 8px;
        text-align: center;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    .section-label {{
        font-size: 0.9em;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .peak-label {{ color: #38a169; }}
    .drop-label {{ color: #e53e3e; }}
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
    }}
</style>
</head>
<body>

<h1>TRIBE Analyzer — Brain Response Report</h1>

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
