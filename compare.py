"""Comparison logic for side-by-side analysis of two TRIBE runs."""

import base64
import json
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

from networks import NETWORK_COLORS, YEO_7_NETWORKS
from persistence import (
    RESULTS_DIR,
    load_all_segment_derived,
    load_analysis_metadata,
)

logger = logging.getLogger(__name__)

COMPARISON_SYSTEM_PROMPT = """\
You are an expert neuroscience communicator comparing two versions of content \
(e.g., before and after editing a video/script/voiceover) for a content creator.

You will receive:
1. A JSON comparison summary showing per-network activation differences between Analysis A and Analysis B
2. A network timeline chart showing both analyses over time
3. A delta chart showing which analysis is stronger per network

Write a clear, actionable comparison report. Structure it as:

## Which Version Wins?

One paragraph: which analysis shows stronger overall brain engagement, and by how much.

## Network-by-Network Breakdown

For each of the 7 networks, one bullet point:
- **[Network name]**: which version is stronger, by how much, and what this means for the content

## Key Differences by Timestamp

Walk through the most significant moments where the two versions diverge. \
Reference specific timestamps and explain what likely causes the difference.

## Recommendation

2-3 sentences: which version should the creator go with, and what specific elements \
from the weaker version could be improved using patterns from the stronger version.

Use plain language. No jargon. Be specific and actionable.
"""


def build_comparison_data(analysis_id_a: str, analysis_id_b: str) -> dict:
    """Build comparison data between two analyses.

    Returns dict with metrics, timelines, and deltas for both analyses.
    """
    meta_a = load_analysis_metadata(analysis_id_a)
    meta_b = load_analysis_metadata(analysis_id_b)
    if not meta_a or not meta_b:
        raise ValueError("One or both analyses not found")

    dir_a = RESULTS_DIR / analysis_id_a
    dir_b = RESULTS_DIR / analysis_id_b

    derived_a = load_all_segment_derived(dir_a)
    derived_b = load_all_segment_derived(dir_b)
    if not derived_a or not derived_b:
        raise ValueError("One or both analyses have incomplete derived data")

    all_na_a, all_pd_a = derived_a
    all_na_b, all_pd_b = derived_b

    # Flatten to single timelines
    flat_a = [act for seg_acts in all_na_a for act in seg_acts]
    flat_b = [act for seg_acts in all_na_b for act in seg_acts]

    overlap = min(len(flat_a), len(flat_b))
    networks = list(YEO_7_NETWORKS.keys())

    # Per-network averages (full duration)
    def avg_by_net(flat):
        return {
            net: round(float(np.mean([t.get(net, 0) for t in flat])), 4)
            for net in networks
        }

    avg_a = avg_by_net(flat_a)
    avg_b = avg_by_net(flat_b)

    total_a = round(sum(sum(t.get(n, 0) for n in networks) for t in flat_a), 2)
    total_b = round(sum(sum(t.get(n, 0) for n in networks) for t in flat_b), 2)

    # Per-second delta (overlapping region only)
    delta_timeline = []
    for i in range(overlap):
        delta = {}
        for net in networks:
            delta[net] = round(flat_a[i].get(net, 0) - flat_b[i].get(net, 0), 4)
        delta_timeline.append(delta)

    # Delta summary
    delta_avg = {
        net: round(avg_a[net] - avg_b[net], 4) for net in networks
    }

    return {
        "label_a": meta_a.get("label", analysis_id_a),
        "label_b": meta_b.get("label", analysis_id_b),
        "input_type_a": meta_a.get("input_type", "unknown"),
        "input_type_b": meta_b.get("input_type", "unknown"),
        "duration_a": len(flat_a),
        "duration_b": len(flat_b),
        "overlap_seconds": overlap,
        "avg_by_network_a": avg_a,
        "avg_by_network_b": avg_b,
        "total_activation_a": total_a,
        "total_activation_b": total_b,
        "delta_avg_by_network": delta_avg,
        "delta_timeline": delta_timeline,
        "timeline_a": flat_a[:overlap],
        "timeline_b": flat_b[:overlap],
        "n_segments_a": len(all_na_a),
        "n_segments_b": len(all_na_b),
    }


def generate_comparison_timeline(
    comparison_data: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a dual-panel timeline chart comparing two analyses."""
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix="tribe_compare_", delete=False,
        )
        output_path = Path(tmp.name)

    overlap = comparison_data["overlap_seconds"]
    timestamps = np.arange(overlap)
    networks = list(YEO_7_NETWORKS.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for net in networks:
        color = NETWORK_COLORS.get(net, "#999")
        label = YEO_7_NETWORKS[net]["layman"]
        vals_a = [comparison_data["timeline_a"][t].get(net, 0) for t in range(overlap)]
        vals_b = [comparison_data["timeline_b"][t].get(net, 0) for t in range(overlap)]
        ax1.plot(timestamps, vals_a, color=color, label=label, linewidth=1.5, alpha=0.85)
        ax2.plot(timestamps, vals_b, color=color, label=label, linewidth=1.5, alpha=0.85)

    ax1.set_title(f"A: {comparison_data['label_a']}", fontsize=12, fontweight="bold")
    ax2.set_title(f"B: {comparison_data['label_b']}", fontsize=12, fontweight="bold")

    for ax in [ax1, ax2]:
        ax.set_ylabel("Activation", fontsize=10)
        ax.grid(True, alpha=0.3)
        # Segment boundaries
        for s in range(20, overlap, 20):
            ax.axvline(x=s, color="#cccccc", linestyle="--", linewidth=1)

    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True)

    fig.suptitle("Brain Network Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def generate_delta_chart(
    comparison_data: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a horizontal bar chart showing per-network deltas (A - B)."""
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix="tribe_delta_", delete=False,
        )
        output_path = Path(tmp.name)

    networks = list(YEO_7_NETWORKS.keys())
    labels = [YEO_7_NETWORKS[n]["layman"] for n in networks]
    deltas = [comparison_data["delta_avg_by_network"][n] for n in networks]
    colors = ["#e53e3e" if d > 0 else "#3182ce" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white", height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta (A minus B)", fontsize=11)
    ax.set_title("Network Activation Delta: A vs B", fontsize=13, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, deltas):
        x = bar.get_width()
        ax.text(
            x + (0.002 if x >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", ha="left" if x >= 0 else "right",
            fontsize=9, color="#333",
        )

    ax.text(0.02, 0.98, "Red = A stronger", transform=ax.transAxes,
            fontsize=9, color="#e53e3e", va="top")
    ax.text(0.02, 0.92, "Blue = B stronger", transform=ax.transAxes,
            fontsize=9, color="#3182ce", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def build_metrics_table(comparison_data: dict) -> list[list]:
    """Build a table of comparison metrics for Gradio Dataframe."""
    cd = comparison_data
    rows = [
        ["Total Activation", f"{cd['total_activation_a']:.2f}",
         f"{cd['total_activation_b']:.2f}",
         f"{cd['total_activation_a'] - cd['total_activation_b']:+.2f}"],
        ["Duration", f"{cd['duration_a']}s", f"{cd['duration_b']}s", "—"],
        ["Segments", str(cd["n_segments_a"]), str(cd["n_segments_b"]), "—"],
    ]

    networks = list(YEO_7_NETWORKS.keys())
    for net in networks:
        label = YEO_7_NETWORKS[net]["layman"]
        val_a = cd["avg_by_network_a"][net]
        val_b = cd["avg_by_network_b"][net]
        delta = cd["delta_avg_by_network"][net]
        rows.append([label, f"{val_a:.4f}", f"{val_b:.4f}", f"{delta:+.4f}"])

    return rows


def generate_comparison_report(
    api_key: str,
    comparison_data: dict,
    timeline_path: str | None = None,
    delta_path: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
) -> str:
    """Generate an LLM comparison report via OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Build summary JSON (exclude raw timelines to save tokens)
    summary = {k: v for k, v in comparison_data.items()
               if k not in ("timeline_a", "timeline_b", "delta_timeline")}

    content = [
        {
            "type": "text",
            "text": (
                f"Compare these two content analyses:\n\n"
                f"**Analysis A**: {comparison_data['label_a']} ({comparison_data['input_type_a']})\n"
                f"**Analysis B**: {comparison_data['label_b']} ({comparison_data['input_type_b']})\n\n"
                f"```json\n{json.dumps(summary, indent=2)}\n```"
            ),
        },
    ]

    for path, label in [(timeline_path, "timeline"), (delta_path, "delta")]:
        if path:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except FileNotFoundError:
                pass

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=3000,
    )

    return response.choices[0].message.content
