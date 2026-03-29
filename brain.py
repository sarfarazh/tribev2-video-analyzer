"""TRIBE v2 brain processing and region aggregation."""

import logging

import numpy as np
import pandas as pd
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain

from networks import classify_region, YEO_7_NETWORKS

logger = logging.getLogger(__name__)

# Global model instances — loaded once at startup
_model: TribeModel | None = None
_plotter: PlotBrain | None = None
_atlas_plotter: PlotBrain | None = None


def load_model(cache_folder: str = "./cache") -> None:
    """Load TRIBE v2 model and plotters. Call once at startup."""
    global _model, _plotter, _atlas_plotter

    logger.info("Loading TRIBE v2 model...")
    _model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=cache_folder,
    )

    logger.info("Initializing plotters...")
    _plotter = PlotBrain(mesh="fsaverage5")
    _atlas_plotter = PlotBrain(
        mesh="fsaverage5", atlas_name="schaefer_2018", atlas_dim=400,
    )

    logger.info("Model and plotters ready.")


def get_model() -> TribeModel:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() first")
    return _model


def get_plotter() -> PlotBrain:
    if _plotter is None:
        raise RuntimeError("Plotter not loaded — call load_model() first")
    return _plotter


def get_atlas_plotter() -> PlotBrain:
    if _atlas_plotter is None:
        raise RuntimeError("Atlas plotter not loaded — call load_model() first")
    return _atlas_plotter


def process_segment(segment_path: str) -> dict:
    """Run TRIBE v2 on a single video segment.

    Args:
        segment_path: Path to the video segment file.

    Returns:
        Dict with keys:
            - preds: np.ndarray of shape (n_timesteps, n_vertices)
            - segments: list of segment objects from TRIBE
            - events: pd.DataFrame of extracted events
    """
    model = get_model()
    events = model.get_events_dataframe(video_path=str(segment_path))
    preds, segments = model.predict(events=events)

    return {
        "preds": preds,
        "segments": segments,
        "events": events,
    }


def _build_network_mapping(atlas_plotter: PlotBrain) -> dict[str, list[int]]:
    """Build a mapping from network names to atlas region indices.

    Uses the Schaefer 400 atlas labels to classify each region into
    one of the 7 Yeo functional networks.

    Returns:
        Dict mapping network name -> list of region indices in the atlas.
    """
    mapping = {name: [] for name in YEO_7_NETWORKS}

    # The atlas plotter should have region labels accessible
    # Try to get labels from the atlas parcellation
    labels = atlas_plotter.atlas_labels if hasattr(atlas_plotter, "atlas_labels") else None

    if labels is not None:
        for idx, label in enumerate(labels):
            network = classify_region(str(label))
            if network is not None:
                mapping[network].append(idx)
    else:
        # Fallback: Schaefer 400 with 7 networks has ~57 regions per network
        # Regions are ordered by network in the standard parcellation
        # LH: 0-199, RH: 200-399
        # Within each hemisphere, regions are grouped by network in order:
        # Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
        network_names = list(YEO_7_NETWORKS.keys())
        regions_per_network = 400 // 7
        for i, name in enumerate(network_names):
            start = i * regions_per_network
            end = start + regions_per_network if i < 6 else 400
            mapping[name] = list(range(start, end))
        logger.warning(
            "Could not get atlas labels — using approximate region mapping"
        )

    return mapping


def aggregate_to_networks(
    preds: np.ndarray,
    atlas_plotter: PlotBrain,
) -> list[dict[str, float]]:
    """Aggregate vertex-level predictions to network-level activations.

    Args:
        preds: Array of shape (n_timesteps, n_vertices).
        atlas_plotter: PlotBrain instance with Schaefer 400 atlas.

    Returns:
        List of dicts (one per timestep), each mapping network name -> mean activation.
    """
    network_mapping = _build_network_mapping(atlas_plotter)

    # If atlas provides a parcellation array mapping vertices to region indices,
    # use it to average vertices within each region first, then average regions
    # within each network. Otherwise, work with raw vertex data.
    parcellation = getattr(atlas_plotter, "parcellation", None)

    per_timestep = []
    for t in range(preds.shape[0]):
        activations = {}
        for network_name, region_indices in network_mapping.items():
            if not region_indices:
                activations[network_name] = 0.0
                continue

            if parcellation is not None:
                # Average vertices belonging to regions in this network
                mask = np.isin(parcellation, region_indices)
                if mask.any():
                    activations[network_name] = float(np.mean(preds[t, mask]))
                else:
                    activations[network_name] = 0.0
            else:
                # Fallback: treat region indices as vertex ranges (approximate)
                n_vertices = preds.shape[1]
                n_regions = 400
                vertices_per_region = n_vertices // n_regions
                vertex_indices = []
                for r in region_indices:
                    start = r * vertices_per_region
                    end = min(start + vertices_per_region, n_vertices)
                    vertex_indices.extend(range(start, end))
                if vertex_indices:
                    activations[network_name] = float(
                        np.mean(preds[t, vertex_indices])
                    )
                else:
                    activations[network_name] = 0.0

        per_timestep.append(activations)

    return per_timestep


def find_peaks_and_drops(
    network_activations: list[dict[str, float]],
    n_peaks: int = 3,
    n_drops: int = 2,
) -> dict:
    """Identify timesteps with highest and lowest overall activation.

    Args:
        network_activations: Per-timestep network activations from aggregate_to_networks.
        n_peaks: Number of peak moments to return.
        n_drops: Number of drop moments to return.

    Returns:
        Dict with 'peaks' and 'drops', each a list of
        (timestep_index, total_activation) tuples.
    """
    totals = [
        sum(act.values()) for act in network_activations
    ]

    sorted_indices = np.argsort(totals)
    peaks = [(int(i), totals[i]) for i in sorted_indices[-n_peaks:][::-1]]
    drops = [(int(i), totals[i]) for i in sorted_indices[:n_drops]]

    return {"peaks": peaks, "drops": drops}
