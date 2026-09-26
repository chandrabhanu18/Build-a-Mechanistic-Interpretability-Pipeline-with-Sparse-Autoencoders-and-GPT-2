#!/usr/bin/env python3
"""
Mechanistic Interpretability Pipeline CLI.
Provides unified commands for SAE feature extraction, correlation analysis, causal steering, and evaluation.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from src.correlation import find_top_features, save_top_features_csv
from src.detector import load_detector_config
from src.extraction import extract_sae_features, load_model_and_sae
from src.steering import evaluate_steering_on_dataset, save_steering_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mi-pipeline")

# Load environment variables
load_dotenv()


def load_dataset_prompts(dataset_path: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load prompts from dataset JSON file, optionally filtered by split."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts", [])
    if split:
        prompts = [p for p in prompts if p.get("split") == split]

    return prompts


def cmd_extract(args: argparse.Namespace) -> None:
    """Execute feature extraction on the training split."""
    logger.info(f"Extracting SAE features from dataset: {args.dataset} (split='{args.split}')")

    train_prompt_items = load_dataset_prompts(args.dataset, split=args.split)
    if not train_prompt_items:
        raise ValueError(f"No prompts found for split '{args.split}' in {args.dataset}")

    prompts = [p["text"] for p in train_prompt_items]
    logger.info(f"Loaded {len(prompts)} prompts for extraction.")

    device = args.device or os.getenv("DEVICE", "cpu")
    model, sae = load_model_and_sae(
        model_name=args.model,
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )

    activations = extract_sae_features(
        model=model,
        sae=sae,
        prompts=prompts,
        batch_size=args.batch_size,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, activations)
    logger.info(f"Saved {activations.shape} activation matrix to {output_path.resolve()}")


def cmd_correlate(args: argparse.Namespace) -> None:
    """Execute correlation analysis and Difference of Means on extracted features."""
    logger.info(f"Loading activations from {args.activations}")
    if not Path(args.activations).exists():
        raise FileNotFoundError(f"Activations file {args.activations} does not exist. Run 'extract' first.")

    activations = np.load(args.activations)

    train_prompt_items = load_dataset_prompts(args.dataset, split=args.split)
    if len(train_prompt_items) != len(activations):
        raise ValueError(
            f"Prompt count ({len(train_prompt_items)}) does not match activation samples ({len(activations)})."
        )

    labels = np.array([p["label"] for p in train_prompt_items], dtype=np.int32)

    logger.info(f"Calculating Difference of Means on {len(labels)} samples ({sum(labels==1)} positive, {sum(labels==0)} negative)...")
    top_features = find_top_features(
        feature_acts=activations,
        labels=labels,
        top_k=args.top_k,
    )

    csv_path = save_top_features_csv(top_features, output_path=args.output)
    logger.info(f"Top {len(top_features)} features saved to {csv_path.resolve()}")

    if top_features:
        best = top_features[0]
        logger.info(f"Best feature: ID={best['feature_id']} | Diff={best['diff_of_means']:+.4f} | "
                    f"Mean(+): {best['mean_positive']:.4f} | Mean(-): {best['mean_negative']:.4f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute causal steering intervention and behavioral shift evaluation on the test split."""
    logger.info(
        f"Evaluating causal steering with feature_id={args.feature_id}, "
        f"multiplier={args.multiplier} on split='{args.split}'"
    )

    test_prompt_items = load_dataset_prompts(args.dataset, split=args.split)
    if not test_prompt_items:
        raise ValueError(f"No prompts found for split '{args.split}' in {args.dataset}")

    detector_config = load_detector_config(args.detector_config)

    device = args.device or os.getenv("DEVICE", "cpu")
    model, sae = load_model_and_sae(
        model_name=args.model,
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )

    metrics = evaluate_steering_on_dataset(
        model=model,
        sae=sae,
        test_prompts=test_prompt_items,
        detector_config=detector_config,
        feature_id=args.feature_id,
        multiplier=args.multiplier,
        max_new_tokens=args.max_new_tokens,
    )

    output_path = save_steering_metrics(metrics, output_path=args.output)
    logger.info(f"Saved evaluation metrics to {output_path.resolve()}")
    print(json.dumps(metrics, indent=2))


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run full pipeline: extract -> correlate -> evaluate."""
    logger.info("Starting complete Mechanistic Interpretability Pipeline run...")

    # Step 1: Extract
    extract_args = argparse.Namespace(
        dataset=args.dataset,
        split="train",
        output=args.output,
        model=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    cmd_extract(extract_args)

    # Step 2: Correlate
    correlate_args = argparse.Namespace(
        activations=args.output,
        dataset=args.dataset,
        split="train",
        output=args.output_csv,
        top_k=args.top_k,
    )
    cmd_correlate(correlate_args)

    # Step 3: Evaluate
    target_feature_id = args.feature_id
    if target_feature_id is None:
        import pandas as pd
        df = pd.read_csv(args.output_csv)
        target_feature_id = int(df.iloc[0]["feature_id"])
        logger.info(f"Auto-selected top correlated feature_id={target_feature_id}")

    eval_args = argparse.Namespace(
        feature_id=target_feature_id,
        multiplier=args.multiplier,
        dataset=args.dataset,
        split="test",
        detector_config=args.detector_config,
        output=args.output_json,
        model=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    cmd_evaluate(eval_args)
    logger.info("Pipeline run finished successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mi-pipeline",
        description="Mechanistic Interpretability Pipeline with Sparse Autoencoders and GPT-2",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract SAE feature activations for dataset")
    p_extract.add_argument("--dataset", default="data/dataset.json", help="Path to labeled dataset JSON")
    p_extract.add_argument("--split", default="train", help="Dataset split to extract (default: train)")
    p_extract.add_argument("--output", default="results/train_activations.npy", help="Output .npy file path")
    p_extract.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt2-small"), help="Model name")
    p_extract.add_argument("--sae-release", default=os.getenv("SAE_RELEASE", "gpt2-small-res-jb"), help="SAE release")
    p_extract.add_argument("--sae-id", default=os.getenv("SAE_ID", "blocks.8.hook_resid_pre"), help="SAE ID")
    p_extract.add_argument("--batch-size", type=int, default=int(os.getenv("EXTRACTION_BATCH_SIZE", "8")), help="Batch size")
    p_extract.add_argument("--device", default=None, help="Compute device (cpu, cuda)")

    # Command: correlate
    p_corr = subparsers.add_parser("correlate", help="Compute Difference of Means and rank correlated features")
    p_corr.add_argument("--activations", default="results/train_activations.npy", help="Activations numpy file")
    p_corr.add_argument("--dataset", default="data/dataset.json", help="Path to labeled dataset JSON")
    p_corr.add_argument("--split", default="train", help="Dataset split (default: train)")
    p_corr.add_argument("--output", default="results/top_features.csv", help="Output CSV file path")
    p_corr.add_argument("--top-k", type=int, default=100, help="Number of top features to save")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate causal steering intervention on test prompts")
    p_eval.add_argument("--feature-id", type=int, default=int(os.getenv("DEFAULT_FEATURE_ID", "14302")), help="Target SAE feature ID")
    p_eval.add_argument("--multiplier", type=float, default=float(os.getenv("DEFAULT_STEERING_MULTIPLIER", "50.0")), help="Intervention strength multiplier")
    p_eval.add_argument("--dataset", default="data/dataset.json", help="Path to labeled dataset JSON")
    p_eval.add_argument("--split", default="test", help="Dataset split (default: test)")
    p_eval.add_argument("--detector-config", default="config/detector.json", help="Path to detector config JSON")
    p_eval.add_argument("--output", default="results/steering_metrics.json", help="Output JSON file path")
    p_eval.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt2-small"), help="Model name")
    p_eval.add_argument("--sae-release", default=os.getenv("SAE_RELEASE", "gpt2-small-res-jb"), help="SAE release")
    p_eval.add_argument("--sae-id", default=os.getenv("SAE_ID", "blocks.8.hook_resid_pre"), help="SAE ID")
    p_eval.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "35")), help="Max new tokens to generate")
    p_eval.add_argument("--device", default=None, help="Compute device (cpu, cuda)")

    # Command: run-all
    p_all = subparsers.add_parser("run-all", help="Run full pipeline: extract -> correlate -> evaluate")
    p_all.add_argument("--dataset", default="data/dataset.json", help="Path to labeled dataset JSON")
    p_all.add_argument("--detector-config", default="config/detector.json", help="Path to detector config JSON")
    p_all.add_argument("--output", default="results/train_activations.npy", help="Activations numpy file")
    p_all.add_argument("--output-csv", default="results/top_features.csv", help="Top features CSV path")
    p_all.add_argument("--output-json", default="results/steering_metrics.json", help="Output metrics JSON")
    p_all.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt2-small"), help="Model name")
    p_all.add_argument("--sae-release", default=os.getenv("SAE_RELEASE", "gpt2-small-res-jb"), help="SAE release")
    p_all.add_argument("--sae-id", default=os.getenv("SAE_ID", "blocks.8.hook_resid_pre"), help="SAE ID")
    p_all.add_argument("--feature-id", type=int, default=None, help="Feature ID (if None, picks top from correlate)")
    p_all.add_argument("--multiplier", type=float, default=50.0, help="Steering multiplier")
    p_all.add_argument("--batch-size", type=int, default=8, help="Extraction batch size")
    p_all.add_argument("--max-new-tokens", type=int, default=35, help="Generation max new tokens")
    p_all.add_argument("--top-k", type=int, default=100, help="Top features count")
    p_all.add_argument("--device", default=None, help="Device")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "correlate":
        cmd_correlate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
