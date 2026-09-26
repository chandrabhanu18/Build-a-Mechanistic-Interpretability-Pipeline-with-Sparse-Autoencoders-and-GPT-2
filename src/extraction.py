"""
Activation Hooking and SAE Feature Extraction Module.
Intercepts internal language model activations and maps them into Sparse Autoencoder latent space.
"""

import logging
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Resolve compute device based on request, environment, and availability.
    """
    if requested_device:
        return torch.device(requested_device)

    env_device = os.getenv("DEVICE", "").strip().lower()
    if env_device:
        return torch.device(env_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_sae(
    model_name: str = "gpt2-small",
    release: str = "gpt2-small-res-jb",
    sae_id: str = "blocks.8.hook_resid_pre",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Any, Any]:
    """
    Load HookedTransformer model and SAELens Sparse Autoencoder.

    Args:
        model_name: Name of the TransformerLens model (default: 'gpt2-small').
        release: SAELens release repository (default: 'gpt2-small-res-jb').
        sae_id: Layer hook identifier (default: 'blocks.8.hook_resid_pre').
        device: Device to place models on.

    Returns:
        Tuple of (HookedTransformer, SAE).
    """
    resolved_device = get_device(str(device) if device else None)
    logger.info(f"Loading {model_name} and SAE {sae_id} on {resolved_device}...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(
        model_name,
        device=resolved_device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(resolved_device),
    )
    sae.eval()

    return model, sae


def extract_sae_features(
    model: Any,
    sae: Any,
    prompts: List[str],
    batch_size: int = 8,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Extracts sparse feature activations for a given dataset of prompts.

    Args:
        model: HookedTransformer model.
        sae: Pre-trained Sparse Autoencoder.
        prompts: List of input prompt strings.
        batch_size: Batch size for tokenization and forward pass.
        device: Compute device.

    Returns:
        np.ndarray of shape (num_prompts, d_sae) representing max feature activations across tokens.
    """
    if not prompts:
        d_sae = getattr(sae.cfg, "d_sae", getattr(sae, "d_sae", 0))
        return np.empty((0, d_sae), dtype=np.float32)

    resolved_device = get_device(str(device) if device else None)
    hook_name = getattr(sae.cfg, "hook_name", "blocks.8.hook_resid_pre")
    all_max_features: List[np.ndarray] = []

    model.eval()
    sae.eval()

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            for prompt in batch_prompts:
                # 1. Run model with cache to intercept target activation tensor
                # activation shape: [1, seq_len, d_model]
                _, cache = model.run_with_cache(
                    prompt,
                    names_filter=[hook_name],
                    return_type=None,
                )
                activations = cache[hook_name]

                # 2. Pass cached activation tensor to sae.encode()
                # Encoded shape: [1, seq_len, d_sae]
                if hasattr(sae, "encode"):
                    feature_acts = sae.encode(activations)
                else:
                    # Generic linear encoding fallback: ReLU(x @ W_enc + b_enc)
                    feature_acts = torch.relu(activations @ sae.W_enc + sae.b_enc)

                # 3. Pool across sequence dimension (max activation over seq_len)
                # Feature acts shape: [1, seq_len, d_sae] -> [d_sae]
                max_acts = feature_acts.squeeze(0).max(dim=0).values

                # Move to CPU numpy array
                all_max_features.append(max_acts.detach().cpu().float().numpy())

    stacked_features = np.stack(all_max_features, axis=0).astype(np.float32)
    logger.info(f"Extracted feature activations matrix with shape {stacked_features.shape}")
    return stacked_features
