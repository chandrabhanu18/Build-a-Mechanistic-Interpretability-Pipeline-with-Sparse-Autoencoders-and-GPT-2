"""
Causal Activation Steering Module.
Injects targeted directional perturbations into the LLM residual stream using SAE decoder vectors.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.detector import detect_behavior, load_detector_config

logger = logging.getLogger(__name__)


def get_steering_hook(
    feature_direction_vector: Union[torch.Tensor, np.ndarray],
    multiplier: float,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Creates a PyTorch forward hook function for HookedTransformer to steer activations.

    Args:
        feature_direction_vector: 1D Tensor or numpy array of shape [d_model] representing
                                  the SAE decoder feature direction.
        multiplier: Scalar strength of intervention (positive for amplification, negative for suppression).

    Returns:
        hook_fn with signature (activations, hook) -> modified_activations
    """
    if not isinstance(feature_direction_vector, torch.Tensor):
        vector_tensor = torch.tensor(feature_direction_vector, dtype=torch.float32)
    else:
        vector_tensor = feature_direction_vector.clone().detach().float()

    def steering_hook(activations: torch.Tensor, hook: Any = None) -> torch.Tensor:
        """
        Applies directional steering: activations + (multiplier * feature_direction_vector)
        """
        # Ensure device and dtype match
        steer_vec = vector_tensor.to(device=activations.device, dtype=activations.dtype)

        # Activations shape: [batch, seq_len, d_model]
        # steer_vec shape: [d_model] -> broadcasts automatically
        return activations + (multiplier * steer_vec)

    return steering_hook


def extract_decoder_vector(sae: Any, feature_id: int) -> torch.Tensor:
    """
    Safely extracts the decoder feature direction vector from an SAE.

    Args:
        sae: Sparse Autoencoder instance.
        feature_id: Index of target feature.

    Returns:
        1D Tensor of shape [d_model].
    """
    if hasattr(sae, "W_dec"):
        w_dec = sae.W_dec
        if hasattr(w_dec, "weight"):
            w_dec = w_dec.weight
        if isinstance(w_dec, torch.Tensor):
            if w_dec.ndim == 2:
                # Typically [d_sae, d_model] or [d_model, d_sae]
                d_sae = getattr(sae.cfg, "d_sae", w_dec.shape[0])
                if w_dec.shape[0] == d_sae:
                    return w_dec[feature_id]
                else:
                    return w_dec[:, feature_id]
            return w_dec[feature_id]

    raise AttributeError("Unable to extract W_dec vector from provided SAE instance.")


def generate_with_steering(
    model: Any,
    sae: Any,
    prompt: str,
    feature_id: int,
    steering_multiplier: float,
    max_new_tokens: int = 35,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
) -> str:
    """
    Generates text with a targeted activation intervention at the SAE hook point.

    Args:
        model: HookedTransformer model.
        sae: Pre-trained Sparse Autoencoder.
        prompt: Input text prompt.
        feature_id: ID of the SAE feature to amplify/suppress.
        steering_multiplier: Intervention strength (0.0 for baseline).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling limit.

    Returns:
        Generated text string.
    """
    hook_name = getattr(sae.cfg, "hook_name", "blocks.8.hook_resid_pre")
    feature_vec = extract_decoder_vector(sae, feature_id)
    steering_hook_fn = get_steering_hook(feature_vec, steering_multiplier)

    # Use model.generate with forward hooks if multiplier != 0, else standard generation
    hooks = [(hook_name, steering_hook_fn)] if abs(steering_multiplier) > 1e-6 else []

    with model.hooks(fwd_hooks=hooks):
        generated_text = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            verbose=False,
            stop_at_eos=True,
        )

    return generated_text


def evaluate_steering_on_dataset(
    model: Any,
    sae: Any,
    test_prompts: List[Dict[str, Any]],
    detector_config: Dict[str, Any],
    feature_id: int,
    multiplier: float,
    max_new_tokens: int = 35,
) -> Dict[str, Any]:
    """
    Evaluates behavioral shift by generating text on test prompts with and without steering.

    Args:
        model: HookedTransformer instance.
        sae: Sparse Autoencoder instance.
        test_prompts: List of prompt dicts from dataset where split == 'test'.
        detector_config: Pre-registered detector configuration.
        feature_id: Target feature ID.
        multiplier: Intervention multiplier.
        max_new_tokens: Maximum new tokens per generation.

    Returns:
        Dict matching results/steering_metrics.json schema.
    """
    baseline_triggers = 0
    steered_triggers = 0
    total_prompts = len(test_prompts)

    if total_prompts == 0:
        return {
            "feature_id": int(feature_id),
            "multiplier": float(multiplier),
            "baseline_behavior_rate": 0.0,
            "steered_behavior_rate": 0.0,
            "absolute_shift": 0.0,
        }

    logger.info(f"Evaluating {total_prompts} test prompts with feature_id={feature_id}, multiplier={multiplier}")

    for idx, prompt_item in enumerate(test_prompts):
        text = prompt_item.get("text", "")

        # 1. Generate baseline (without steering)
        baseline_gen = generate_with_steering(
            model=model,
            sae=sae,
            prompt=text,
            feature_id=feature_id,
            steering_multiplier=0.0,
            max_new_tokens=max_new_tokens,
        )
        baseline_detected = detect_behavior(baseline_gen, detector_config)
        if baseline_detected:
            baseline_triggers += 1

        # 2. Generate with causal steering hook
        steered_gen = generate_with_steering(
            model=model,
            sae=sae,
            prompt=text,
            feature_id=feature_id,
            steering_multiplier=multiplier,
            max_new_tokens=max_new_tokens,
        )
        steered_detected = detect_behavior(steered_gen, detector_config)
        if steered_detected:
            steered_triggers += 1

        logger.debug(
            f"Prompt [{idx+1}/{total_prompts}]: '{text[:40]}...' | "
            f"Baseline: {baseline_detected} | Steered: {steered_detected}"
        )

    baseline_rate = float(baseline_triggers / total_prompts)
    steered_rate = float(steered_triggers / total_prompts)
    absolute_shift = float(steered_rate - baseline_rate)

    metrics = {
        "feature_id": int(feature_id),
        "multiplier": float(multiplier),
        "baseline_behavior_rate": round(baseline_rate, 4),
        "steered_behavior_rate": round(steered_rate, 4),
        "absolute_shift": round(absolute_shift, 4),
    }

    logger.info(
        f"Intervention Results -> Baseline: {baseline_rate:.2%}, "
        f"Steered: {steered_rate:.2%}, Shift: {absolute_shift:+.2%}"
    )
    return metrics


def save_steering_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path] = "results/steering_metrics.json",
) -> Path:
    """
    Save steering metrics dictionary to JSON file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved steering metrics to {out_file.resolve()}")
    return out_file
