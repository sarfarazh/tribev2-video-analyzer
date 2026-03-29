"""Build timestamped JSON summary from TRIBE v2 predictions."""

import json
from pathlib import Path

import numpy as np


def build_segment_summary(
    network_activations: list[dict[str, float]],
    events_df,
    time_offset: float,
    segment_index: int,
    peaks_drops: dict,
) -> dict:
    """Build JSON-serializable summary for one video segment.

    Args:
        network_activations: Per-timestep network activations.
        events_df: Events DataFrame from TRIBE for this segment.
        time_offset: Start time of this segment in the original video.
        segment_index: Index of this segment (0-based).
        peaks_drops: Dict with 'peaks' and 'drops' from find_peaks_and_drops.

    Returns:
        Dict representing this segment's analysis.
    """
    peak_indices = {t for t, _ in peaks_drops["peaks"]}
    drop_indices = {t for t, _ in peaks_drops["drops"]}

    # Extract transcript words per timestep
    word_events = events_df[events_df["type"] == "Word"] if "type" in events_df.columns else None

    timesteps = []
    for t, activations in enumerate(network_activations):
        abs_time = time_offset + t
        time_label = f"{int(abs_time // 60)}:{int(abs_time % 60):02d}"

        # Find words spoken at this timestep
        words = []
        if word_events is not None and not word_events.empty:
            mask = (
                (word_events["start"] >= t)
                & (word_events["start"] < t + 1)
            )
            words = word_events.loc[mask, "text"].tolist()

        dominant = max(activations, key=activations.get) if activations else None

        timesteps.append({
            "time": time_label,
            "abs_seconds": abs_time,
            "networks": {k: round(v, 4) for k, v in activations.items()},
            "dominant_network": dominant,
            "transcript_words": words,
            "is_peak": t in peak_indices,
            "is_drop": t in drop_indices,
        })

    # Segment-level averages
    all_networks = list(network_activations[0].keys()) if network_activations else []
    avg_by_network = {}
    for net in all_networks:
        values = [act[net] for act in network_activations]
        avg_by_network[net] = round(float(np.mean(values)), 4)

    dominant_overall = max(avg_by_network, key=avg_by_network.get) if avg_by_network else None

    end_time = time_offset + len(network_activations)
    time_range = (
        f"{int(time_offset // 60)}:{int(time_offset % 60):02d}"
        f" - {int(end_time // 60)}:{int(end_time % 60):02d}"
    )

    return {
        "segment_index": segment_index,
        "time_range": time_range,
        "timesteps": timesteps,
        "segment_summary": {
            "avg_activation_by_network": avg_by_network,
            "peak_moments": [
                {"timestep": t, "time": timesteps[t]["time"], "activation": round(v, 4)}
                for t, v in peaks_drops["peaks"]
                if t < len(timesteps)
            ],
            "drop_moments": [
                {"timestep": t, "time": timesteps[t]["time"], "activation": round(v, 4)}
                for t, v in peaks_drops["drops"]
                if t < len(timesteps)
            ],
            "dominant_network": dominant_overall,
        },
    }


def build_full_summary(
    segment_summaries: list[dict],
    video_duration: float,
    segment_duration: int = 20,
) -> dict:
    """Combine segment summaries into a full video analysis.

    Args:
        segment_summaries: List of dicts from build_segment_summary.
        video_duration: Total video duration in seconds.
        segment_duration: Duration of each segment in seconds.

    Returns:
        Complete JSON-serializable analysis dict.
    """
    # Find global peaks and drops across all segments
    all_timesteps = []
    for seg in segment_summaries:
        all_timesteps.extend(seg["timesteps"])

    totals = [
        (ts, sum(ts["networks"].values()))
        for ts in all_timesteps
    ]
    totals_sorted = sorted(totals, key=lambda x: x[1], reverse=True)

    global_peaks = [
        {"time": ts["time"], "abs_seconds": ts["abs_seconds"],
         "activation": round(val, 4), "dominant_network": ts["dominant_network"]}
        for ts, val in totals_sorted[:5]
    ]
    global_drops = [
        {"time": ts["time"], "abs_seconds": ts["abs_seconds"],
         "activation": round(val, 4), "dominant_network": ts["dominant_network"]}
        for ts, val in totals_sorted[-3:]
    ]

    return {
        "video_duration_seconds": round(video_duration, 1),
        "segment_duration_seconds": segment_duration,
        "total_segments": len(segment_summaries),
        "segments": segment_summaries,
        "overall": {
            "global_peaks": global_peaks,
            "global_drops": global_drops,
        },
    }


def save_summary(summary: dict, output_path: str | Path) -> str:
    """Save the analysis summary to a JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return str(output_path)
