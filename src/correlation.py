"""
Correlation Analysis Engine.
Computes statistical correlation and Difference of Means between sparse SAE feature activations
and behavioral target labels.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_top_features(
    feature_acts: np.ndarray,
    labels: np.ndarray,
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """
    Identifies top correlating features using Difference of Means.

    Args:
        feature_acts: Shape (num_prompts, num_features) np.ndarray of feature activations.
        labels: Shape (num_prompts,) binary labels (1 for trigger, 0 for non-trigger).
        top_k: Number of top features to return.

    Returns:
        List of dicts containing feature_id, diff_of_means, mean_positive, mean_negative.
    """
    feature_acts = np.asarray(feature_acts, dtype=np.float64)
    labels = np.asarray(labels).flatten()

    if len(feature_acts) != len(labels):
        raise ValueError(
            f"Dimension mismatch: feature_acts has {len(feature_acts)} samples, "
            f"labels has {len(labels)} samples."
        )

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    if not np.any(pos_mask):
        raise ValueError("Cannot calculate Difference of Means: No positive samples found in labels.")
    if not np.any(neg_mask):
        raise ValueError("Cannot calculate Difference of Means: No negative samples found in labels.")

    # Calculate mean activation for positive and negative prompt sets
    mean_positive = np.mean(feature_acts[pos_mask], axis=0)
    mean_negative = np.mean(feature_acts[neg_mask], axis=0)
    diff_of_means = mean_positive - mean_negative

    # Sort features by the absolute magnitude of this difference
    abs_diff = np.abs(diff_of_means)
    sorted_indices = np.argsort(-abs_diff)

    num_features_to_return = min(top_k, len(sorted_indices))
    top_indices = sorted_indices[:num_features_to_return]

    top_features: List[Dict[str, Any]] = []
    for idx in top_indices:
        top_features.append({
            "feature_id": int(idx),
            "diff_of_means": float(diff_of_means[idx]),
            "mean_positive": float(mean_positive[idx]),
            "mean_negative": float(mean_negative[idx]),
        })

    logger.info(f"Identified top {len(top_features)} features via Difference of Means.")
    return top_features


def save_top_features_csv(
    top_features: List[Dict[str, Any]],
    output_path: Union[str, Path] = "results/top_features.csv",
) -> Path:
    """
    Save top features list to CSV file with required headers.

    Required Headers:
        feature_id, diff_of_means, mean_positive, mean_negative

    Args:
        top_features: List of feature dicts.
        output_path: Output CSV file path.

    Returns:
        Path to written CSV file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(top_features)
    columns = ["feature_id", "diff_of_means", "mean_positive", "mean_negative"]

    # Ensure correct columns and ordering
    if not df.empty:
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(out_file, index=False)
    logger.info(f"Saved {len(df)} top features to {out_file.resolve()}")
    return out_file
