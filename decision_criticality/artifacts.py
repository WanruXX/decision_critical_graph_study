"""Portable study inputs and frozen model verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .example import pack_observations
from .models import GraphActionModel, ModelConfig, predict_observations


ASSETS = Path(__file__).parent / "assets"
FAMILIES = ("baseline", "uniform", "repeated", "independent")
SEEDS = (42, 73, 101)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(verify=True):
    manifest = load_json(ASSETS / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if file_hash(ASSETS / name) != expected:
            raise ValueError(f"study artifact checksum mismatch: {name}")
    models = {}
    for family in FAMILIES:
        for seed in SEEDS:
            name = f"{family}_{seed}"
            saved = torch.load(ASSETS / "models" / f"{name}.pt", weights_only=True, map_location="cpu")
            if saved["input_protocol"] != "normalize_evidence_v1" or saved["seed"] != seed:
                raise ValueError("unsupported checkpoint input protocol or seed")
            model = GraphActionModel(ModelConfig(**saved["config"]))
            model.load_state_dict(saved["state_dict"])
            models[name] = model
    if verify:
        fixture = load_json(ASSETS / "replay.json")
        actual = predict_observations(fixture["observations"], models, pack_observations)
        errors = {name: float(np.abs(values - np.asarray(fixture["probabilities"][name])).max())
                  for name, values in actual.items()}
        if max(errors.values()) > 1e-6:
            raise ValueError(f"checkpoint replay exceeds numerical tolerance: {errors}")
    return models