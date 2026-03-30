"""Brain network definitions mapping Schaefer 400 atlas regions to functional networks."""

# Schaefer 400 atlas uses a naming convention where each region label contains
# the network name. These are the 7 Yeo networks used in the atlas.
# Reference: Schaefer et al. 2018, Yeo et al. 2011

YEO_7_NETWORKS = {
    "visual": {
        "atlas_keywords": ["Vis"],
        "description": "Visual processing — responds to scene changes, motion, faces, colors",
        "layman": "What you see",
    },
    "somatomotor": {
        "atlas_keywords": ["SomMot"],
        "description": "Motor and sensory processing — body movement, touch, physical sensation",
        "layman": "Physical sensation and movement",
    },
    "dorsal_attention": {
        "atlas_keywords": ["DorsAttn"],
        "description": "Top-down attention — voluntary focus, tracking objects, spatial awareness",
        "layman": "Focused attention",
    },
    "ventral_attention": {
        "atlas_keywords": ["SalVentAttn"],
        "description": "Salience and surprise detection — noticing unexpected events, alertness",
        "layman": "Surprise and alertness",
    },
    "limbic": {
        "atlas_keywords": ["Limbic"],
        "description": "Emotion and memory — emotional reactions, reward, motivation",
        "layman": "Emotional response",
    },
    "frontoparietal": {
        "atlas_keywords": ["Cont"],
        "description": "Executive control — decision-making, working memory, problem-solving",
        "layman": "Thinking and decision-making",
    },
    "default_mode": {
        "atlas_keywords": ["Default"],
        "description": "Self-referential thought — mind wandering, narrative processing, empathy",
        "layman": "Personal connection and storytelling",
    },
}


NETWORK_COLORS = {
    "visual": "#1f77b4",
    "somatomotor": "#ff7f0e",
    "dorsal_attention": "#2ca02c",
    "ventral_attention": "#d62728",
    "limbic": "#9467bd",
    "frontoparietal": "#8c564b",
    "default_mode": "#e377c2",
}


def classify_region(region_label: str) -> str | None:
    """Map a Schaefer atlas region label to a functional network name.

    Args:
        region_label: Full region label from Schaefer 400 atlas
            (e.g., "7Networks_LH_Vis_1")

    Returns:
        Network name key from YEO_7_NETWORKS, or None if unmatched.
    """
    for network_name, info in YEO_7_NETWORKS.items():
        for keyword in info["atlas_keywords"]:
            if keyword in region_label:
                return network_name
    return None
