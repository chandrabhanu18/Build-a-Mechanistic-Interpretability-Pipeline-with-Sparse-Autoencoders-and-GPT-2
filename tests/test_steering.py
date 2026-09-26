"""
Unit tests for Mechanistic Interpretability Pipeline.
Verifies core tensor mathematics, hook modifications, correlation analysis, and detector logic.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.correlation import find_top_features, save_top_features_csv
from src.detector import detect_behavior, load_detector_config
from src.steering import get_steering_hook, save_steering_metrics


class TestSteeringHook:
    """Tests for tensor steering hook mathematics."""

    def test_steering_hook_tensor_addition(self):
        """
        Verify that applying the steering hook adds exactly (multiplier * feature_direction_vector)
        to a mock activation tensor.
        """
        batch_size = 2
        seq_len = 4
        d_model = 768
        multiplier = 35.5

        # Create zero activation tensor [batch, seq_len, d_model]
        mock_activations = torch.zeros((batch_size, seq_len, d_model), dtype=torch.float32)

        # Create deterministic feature direction vector [d_model]
        feature_vector = torch.randn(d_model, dtype=torch.float32)

        # Create steering hook
        hook_fn = get_steering_hook(feature_vector, multiplier)

        # Apply hook (passing None as hook parameter)
        modified_activations = hook_fn(mock_activations, hook=None)

        # Expected result: each token position is exactly multiplier * feature_vector
        expected_addition = multiplier * feature_vector
        expected_tensor = expected_addition.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, d_model)

        assert modified_activations.shape == mock_activations.shape
        assert torch.allclose(modified_activations, expected_tensor, atol=1e-5)

    def test_steering_hook_with_nonzero_initial_activations(self):
        """Verify steering hook on non-zero activations."""
        d_model = 64
        activations = torch.ones((1, 3, d_model), dtype=torch.float32) * 2.0
        feature_vector = torch.ones(d_model, dtype=torch.float32) * 0.5
        multiplier = 10.0

        hook_fn = get_steering_hook(feature_vector, multiplier)
        out = hook_fn(activations, None)

        # 2.0 + (10.0 * 0.5) = 7.0
        assert torch.allclose(out, torch.ones_like(activations) * 7.0)

    def test_steering_hook_negative_multiplier(self):
        """Verify suppression behavior (negative multiplier)."""
        d_model = 32
        activations = torch.ones((1, 1, d_model), dtype=torch.float32) * 5.0
        feature_vector = torch.ones(d_model, dtype=torch.float32) * 2.0
        multiplier = -2.0

        hook_fn = get_steering_hook(feature_vector, multiplier)
        out = hook_fn(activations, None)

        # 5.0 + (-2.0 * 2.0) = 1.0
        assert torch.allclose(out, torch.ones_like(activations) * 1.0)


class TestCorrelationAnalysis:
    """Tests for Difference of Means statistical ranking."""

    def test_find_top_features_synthetic(self):
        """Test difference of means ranking on synthetic activations."""
        # 4 samples, 5 features
        # Features 0, 1, 2, 3, 4
        # Positive samples (labels=1): samples 0, 1
        # Negative samples (labels=0): samples 2, 3
        feature_acts = np.array([
            [10.0, 0.0, 5.0, 1.0, 2.0],  # pos 1
            [8.0,  0.0, 5.0, 3.0, 2.0],  # pos 2
            [0.0,  0.0, 5.0, 8.0, 2.0],  # neg 1
            [0.0,  0.0, 5.0, 6.0, 2.0],  # neg 2
        ], dtype=np.float32)

        labels = np.array([1, 1, 0, 0])

        top_features = find_top_features(feature_acts, labels, top_k=3)

        assert len(top_features) == 3
        # Feature 0: mean(pos)=9.0, mean(neg)=0.0 -> diff = +9.0
        # Feature 3: mean(pos)=2.0, mean(neg)=7.0 -> diff = -5.0 (abs diff = 5.0)
        # Feature 1, 2, 4: diff = 0.0
        assert top_features[0]["feature_id"] == 0
        assert pytest.approx(top_features[0]["diff_of_means"], abs=1e-4) == 9.0
        assert pytest.approx(top_features[0]["mean_positive"], abs=1e-4) == 9.0
        assert pytest.approx(top_features[0]["mean_negative"], abs=1e-4) == 0.0

        assert top_features[1]["feature_id"] == 3
        assert pytest.approx(top_features[1]["diff_of_means"], abs=1e-4) == -5.0

    def test_save_top_features_csv(self, tmp_path):
        """Verify CSV saving and schema."""
        features = [
            {"feature_id": 14302, "diff_of_means": 4.52, "mean_positive": 5.12, "mean_negative": 0.60},
            {"feature_id": 1024, "diff_of_means": -3.21, "mean_positive": 0.10, "mean_negative": 3.31},
        ]
        csv_file = tmp_path / "top_features.csv"
        saved_path = save_top_features_csv(features, output_path=csv_file)

        assert saved_path.exists()
        import pandas as pd
        df = pd.read_csv(saved_path)
        assert list(df.columns) == ["feature_id", "diff_of_means", "mean_positive", "mean_negative"]
        assert len(df) == 2
        assert int(df.iloc[0]["feature_id"]) == 14302


class TestBehaviorDetector:
    """Tests for regex-based behavior detector."""

    @pytest.fixture
    def detector_config(self):
        return load_detector_config("config/detector.json")

    def test_detect_hedging_positive(self, detector_config):
        """Test true positive detection on hedging sentences."""
        samples = [
            "It really depends on the severity and duration of the symptoms.",
            "You should consult a doctor before taking any medication.",
            "I'm not entirely sure, but this could be a minor infection.",
            "Please speak with a medical professional immediately.",
            "Potential causes include dehydration and muscle strain.",
        ]
        for s in samples:
            assert detect_behavior(s, detector_config) is True, f"Failed on: {s}"

    def test_detect_hedging_negative(self, detector_config):
        """Test true negative detection on direct non-hedging answers."""
        samples = [
            "The boiling point of water is 100 degrees Celsius.",
            "Paris is the capital of France.",
            "12 multiplied by 8 is 96.",
            "William Shakespeare wrote Romeo and Juliet.",
            "The chemical symbol for gold is Au.",
        ]
        for s in samples:
            assert detect_behavior(s, detector_config) is False, f"Failed on: {s}"


class TestSchemasAndConfigs:
    """Validate project configuration files and datasets."""

    def test_dataset_json_validity(self):
        """Verify data/dataset.json schema and requirements."""
        dataset_path = Path("data/dataset.json")
        assert dataset_path.exists(), "data/dataset.json must exist"

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "behavior_name" in data
        assert "prompts" in data
        prompts = data["prompts"]

        # Verification: length between 40 and 60
        assert 40 <= len(prompts) <= 60, f"Expected 40-60 prompts, got {len(prompts)}"

        # Check prompt schema and split/label distributions
        train_prompts = [p for p in prompts if p.get("split") == "train"]
        test_prompts = [p for p in prompts if p.get("split") == "test"]
        pos_prompts = [p for p in prompts if p.get("label") == 1]
        neg_prompts = [p for p in prompts if p.get("label") == 0]

        assert len(train_prompts) > 0, "Missing train split prompts"
        assert len(test_prompts) > 0, "Missing test split prompts"
        assert len(pos_prompts) > 0, "Missing positive label prompts"
        assert len(neg_prompts) > 0, "Missing negative label prompts"

        for p in prompts:
            assert "id" in p and isinstance(p["id"], str)
            assert "text" in p and isinstance(p["text"], str) and len(p["text"]) > 0
            assert "label" in p and p["label"] in (0, 1)
            assert "split" in p and p["split"] in ("train", "test")

    def test_detector_config_validity(self):
        """Verify config/detector.json schema."""
        config = load_detector_config("config/detector.json")
        assert "behavior_target" in config
        assert "detection_type" in config
        assert "rules" in config
        assert len(config["rules"]) >= 1

    def test_submission_json_validity(self):
        """Verify submission.json schema."""
        sub_path = Path("submission.json")
        assert sub_path.exists(), "submission.json must exist"

        with open(sub_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        assert "target_behavior" in sub
        assert "best_candidate_feature_id" in sub
        assert "optimal_multiplier" in sub
        assert isinstance(sub["best_candidate_feature_id"], int)
        assert isinstance(sub["optimal_multiplier"], (int, float))

    def test_env_example_validity(self):
        """Verify .env.example content."""
        env_path = Path(".env.example")
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        assert "DEVICE" in content


class TestSteeringMetricsPersistence:
    """Test saving steering metrics."""

    def test_save_steering_metrics(self, tmp_path):
        metrics = {
            "feature_id": 14302,
            "multiplier": 50.0,
            "baseline_behavior_rate": 0.40,
            "steered_behavior_rate": 0.85,
            "absolute_shift": 0.45,
        }
        out_json = tmp_path / "steering_metrics.json"
        saved = save_steering_metrics(metrics, out_json)

        assert saved.exists()
        with open(saved, "r") as f:
            loaded = json.load(f)

        assert loaded["feature_id"] == 14302
        assert loaded["multiplier"] == 50.0
        assert loaded["absolute_shift"] == 0.45
