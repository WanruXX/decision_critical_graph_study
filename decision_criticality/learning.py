"""Matched exposure controls for learned evidence interpretation."""

import argparse
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .artifacts import ASSETS, SEEDS, file_hash, load_json, load_models, write_json
from .example import CONDITIONS, observation_variant, pack_observations, posterior
from .models import predict


def build_dataset(instances, rows=None):
    views = {(instance["game_id"], view["observer"]): view for instance in instances for view in instance["views"]}
    if rows is None:
        rows = [(instance["game_id"], view["observer"], condition) for instance in instances
                for view in instance["views"] for condition in CONDITIONS]
    if not rows:
        raise ValueError("nonempty training rows required")
    observations, exact, cache = [], [], {}
    for instance, observer, condition in rows:
        key = instance, observer, condition
        if key not in cache:
            view = observation_variant(views[(instance, observer)], condition)
            values = posterior(view)
            candidates = [player for player in range(6) if player != observer]
            cache[key] = view, tuple(values[player] for player in candidates)
        view, values = cache[key]
        observations.append(view)
        exact.append(values)
    targets = torch.tensor([[float(value) for value in values] for values in exact])
    near = torch.tensor([0 < sorted(values, reverse=True)[0] - sorted(values, reverse=True)[1] <= Fraction(1, 50)
                         for values in exact], dtype=torch.bool)
    return {"packed": pack_observations(observations), "targets": targets, "exact": exact, "near": near,
            "instance_ids": {row[0] for row in rows}}


def validation_score(model, data):
    probabilities = predict(model, data["packed"])
    actions = probabilities.argmax(-1)
    regret = np.asarray([float(max(values) - values[action]) for values, action in zip(data["exact"], actions)])
    targets = np.asarray([[float(value) for value in values] for values in data["exact"]])
    return {"near_regret": float(regret[data["near"].numpy()].mean()) if data["near"].any() else None,
            "overall_regret": float(regret.mean()), "posterior_mae": float(np.abs(targets - probabilities).mean()),
            "cross_entropy": float(-(targets * np.log(np.maximum(probabilities, 1e-12))).sum(-1).mean())}


def fit_model(template, train, validation, seed, epochs=10):
    if epochs <= 0 or train["instance_ids"].intersection(validation["instance_ids"]):
        raise ValueError("positive epoch count and disjoint training/validation instances required")
    model = deepcopy(template)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    initial = validation_score(model, validation)
    best, state, best_epoch = initial, deepcopy(model.state_dict()), 0
    history, steps, exposures = [{"epoch": 0, **initial}], 0, 0

    def key(score):
        return (score["near_regret"] if score["near_regret"] is not None else score["overall_regret"],
                score["overall_regret"], score["cross_entropy"])

    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(train["targets"]))
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), 64):
            selected = torch.as_tensor(order[start:start + 64], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tuple(values[selected] for values in train["packed"]))
            loss = -(train["targets"][selected] * logits.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
            steps += 1
            exposures += int(train["near"][selected].sum())
        current = validation_score(model, validation)
        history.append({"epoch": epoch, "training_loss": total_loss / len(order), **current})
        if current["cross_entropy"] <= initial["cross_entropy"] + 0.01 and current["posterior_mae"] <= 0.03 and key(current) < key(best):
            best, state, best_epoch = current, deepcopy(model.state_dict()), epoch
    model.load_state_dict(state)
    return model, {"best_epoch": best_epoch, "validation": best, "history": history, "optimizer_steps": steps,
                   "row_presentations": epochs * len(train["targets"]), "near_presentations": exposures}


def run_training(output_dir, epochs=10):
    if epochs <= 0:
        raise ValueError("positive epoch count required")
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_json(ASSETS / "training.json")
    models = load_models()
    families = ("uniform", "repeated", "independent")
    train_instances = inputs["baseline_train"] + inputs["independent_train"]
    validation_instances = inputs["validation"]["general"] + inputs["validation"]["targeted"]
    protocol = {"families": list(families), "training_seeds": list(SEEDS), "epochs": epochs,
                "training_data_sha256": file_hash(ASSETS / "training.json"),
                "artifact_manifest_sha256": file_hash(ASSETS / "manifest.json"),
                "checkpoint_rule": "validation near regret, overall regret, CE; CE drift <= .01; MAE <= .03; epoch zero eligible",
                "loss": "soft-target cross entropy", "learning_rate": 0.0003, "weight_decay": 0.0001,
                "batch_size": 64, "gradient_clip": 5,
                "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")}}
    write_json(output / "protocol.json", protocol)
    datasets = {family: build_dataset(train_instances, inputs["training_slots"][family]) for family in families}
    validation = build_dataset(validation_instances, inputs["validation_rows"])
    if datasets["repeated"]["near"].sum() != datasets["independent"]["near"].sum():
        raise AssertionError("enriched arms must have identical near exposure")
    fits = {}
    for seed in SEEDS:
        saved = torch.load(ASSETS / "models" / f"baseline_{seed}.pt", weights_only=True)
        for family in families:
            name = f"{family}_{seed}"
            model, fit = fit_model(models[f"baseline_{seed}"], datasets[family], validation, seed, epochs)
            torch.save({"config": saved["config"], "state_dict": model.state_dict(), "seed": seed,
                        "family": family, "input_protocol": "normalize_evidence_v1"}, output / f"{name}.pt")
            fits[name] = fit
            print(f"{name}: selected epoch {fit['best_epoch']}", flush=True)
    write_json(output / "fits.json", fits)
    print(output.resolve(), flush=True)
    return output, fits


def main():
    parser = argparse.ArgumentParser(description="Retrain matched evidence-exposure controls from frozen starting models")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    arguments = parser.parse_args()
    run_training(arguments.output_dir, arguments.epochs)


if __name__ == "__main__":
    main()