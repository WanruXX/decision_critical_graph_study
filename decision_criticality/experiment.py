"""Evaluate consequences of controlled evidence edits."""

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import platform

import numpy as np
import torch

from .artifacts import ASSETS, FAMILIES, SEEDS, file_hash, load_json, load_models, write_json
from .contracts import consequence_readout, evaluate_decision
from .evaluation import classification_metrics, paired_intervals, select_threshold
from .example import build_pairs, instance_signature, pack_observations, reference_posterior, sample_instances
from .models import GraphActionModel, ModelConfig, predict_observations


def evaluate_pairs(observations, pairs, probabilities, thresholds, data_seeds):
    exact = [tuple(Fraction(value) for value in item["probabilities"]) for item in observations]
    rows = {}
    for method, values in probabilities.items():
        seed = method.rsplit("_", 1)[1]
        baseline = probabilities[f"baseline_{seed}"]
        selected = []
        for pair in pairs:
            before, after = pair["reference_index"], pair["edited_index"]
            candidates = [player for player in range(6) if player != pair["observer"]]
            reference = tuple(exact[before][player] for player in candidates)
            edited = tuple(exact[after][player] for player in candidates)
            prediction = consequence_readout(values[before], values[after])
            detail = evaluate_decision(reference, edited, prediction["action"], prediction["initial_action"])
            score = prediction["predicted_regret"]
            common = int(baseline[after].argmax())
            common_exact = float(max(reference) - reference[common])
            common_score = float(values[before].max() - values[before][common])
            ordered = sorted(edited, reverse=True)
            protected = pair["edit"] == "remove_one_copy" or (
                pair["edit"] == "changed_author" and pair["root"] in observations[before]["view"]["audit"])
            row = {**pair, **detail, "method": method, "data_seed": data_seeds[pair["game_id"]],
                   "player_action": candidates[prediction["action"]], "predicted_regret": score,
                   "regret_error": abs(score - detail["reference_regret"]), "common_action": common,
                   "common_action_regret": common_exact, "common_action_error": abs(common_score - common_exact),
                   "near_tie": 0 < ordered[0] - ordered[1] <= Fraction(1, 50),
                   "protected": protected, "suboptimal_action": detail["edited_regret"] > 0}
            if thresholds is not None:
                for column, head in enumerate(("any_harm", "material_harm")):
                    row[f"{head}_error"] = int((score >= thresholds[method][column]) != bool(row[head]))
            selected.append(row)
        rows[method] = selected
    return rows


def calibrate(models, validation_instances):
    observations, pairs = build_pairs(validation_instances)
    values = predict_observations(observations, models, pack_observations)
    seeds = {instance["game_id"]: instance.get("data_seed", 0) for instance in validation_instances}
    rows = evaluate_pairs(observations, pairs, values, None, seeds)
    return {method: [select_threshold([row[head] for row in group], [row["predicted_regret"] for row in group])
                     for head in ("any_harm", "material_harm")] for method, group in rows.items()}


def summarize(rows, thresholds):
    result = {}
    means = ("regret_error", "common_action_error", "edited_regret", "reference_regret", "incremental_reference_regret",
             "suboptimal_action", "action_changed", "any_harm_error", "material_harm_error",
             "minimum_optimal_regret", "maximum_optimal_regret", "possible_any_harm", "unavoidable_any_harm",
             "possible_material_harm", "unavoidable_material_harm")
    for method, group in rows.items():
        scopes = {"all": group}
        for field in ("near_tie", "edited_tie", "policy_dependent", "protected"):
            scopes[field] = [row for row in group if row[field]]
        for field in ("role", "context", "edit"):
            for value in sorted({row[field] for row in group}):
                scopes[f"{field}:{value}"] = [row for row in group if row[field] == value]
        result[method] = {}
        for scope, selected in scopes.items():
            if not selected:
                continue
            metrics = {"pairs": len(selected), "games": len({row["game_id"] for row in selected})}
            metrics.update({field: float(np.mean([row[field] for row in selected])) for field in means})
            for column, head in enumerate(("any_harm", "material_harm")):
                metrics[head] = classification_metrics([row[head] for row in selected],
                                                       [row["predicted_regret"] for row in selected], thresholds[method][column])
            result[method][scope] = metrics
    return result


def compare_reference(summary, tolerance=1e-10):
    saved = load_json(ASSETS / "reference.json")
    fields = {"regret_error": "regret_mae", "common_action_error": "common_action_absolute_error",
              "edited_regret": "edited_posterior_regret", "reference_regret": "own_policy_regret",
              "incremental_reference_regret": "incremental_reference_regret"}
    if summary["games"] != saved["games"] or summary["pairs"] != saved["pairs"] or summary["observations"] != saved["observations"]:
        raise AssertionError("published evaluation denominator mismatch")
    maximum = 0.0
    for method, scopes in summary["metrics"].items():
        for field, previous in fields.items():
            maximum = max(maximum, abs(scopes["all"][field] - saved["metrics"][method]["all"][previous]))
        for head in ("any_harm", "material_harm"):
            maximum = max(maximum, abs(scopes["all"][head]["f1"] - saved["metrics"][method]["all"][head]["f1"]))
    if maximum > tolerance:
        raise AssertionError(f"published metrics differ by {maximum}; tolerance {tolerance}")
    return maximum


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def report(summary):
    lines = ["# Evidence Edits and Downstream Decisions", "", f"Mode: {summary['mode']}. "
             f"Instances: {summary['games']}; paired cases: {summary['pairs']}; exact-checked observations: {summary['observations']}.", "",
             "Targets evaluate each model's executed action. Common-action MAE holds the baseline's action fixed. "
             "Reference loss and edited-evidence suboptimality are distinct. Bounds range over oracle edited-optimal actions only.", "",
             "| Family | Regret MAE | Common-action MAE | Edited regret | Reference regret | Any F1 | Material F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family in FAMILIES:
        metrics = [summary["metrics"][f"{family}_{seed}"]["all"] for seed in SEEDS]
        values = [np.mean([entry[field] for entry in metrics]) for field in (
            "regret_error", "common_action_error", "edited_regret", "reference_regret")]
        values += [np.mean([entry[head]["f1"] for entry in metrics]) for head in ("any_harm", "material_harm")]
        lines.append(f"| {family} | " + " | ".join(f"{value:.8f}" for value in values) + " |")
    lines.extend(["", "## Paired Equal-Instance Contrasts", "", "Differences average fixed training seeds inside whole instances. "
                  "Intervals resample whole instances within generation-seed strata, conditional on models and thresholds. "
                  "Secondary comparisons are unadjusted. Case-weighted means above can differ from equal-instance contrasts.", "",
                  "| Scope | Contrast | Metric | Difference | 95% interval |", "| --- | --- | --- | ---: | --- |"])
    for row in summary["paired_intervals"]:
        if row["scope"] not in ("all", "near_tie") or row["contrast"] not in ("independent_minus_baseline", "independent_minus_repeated"):
            continue
        if row["metric"] not in ("regret_error", "common_action_error", "edited_regret", "reference_regret"):
            continue
        lines.append(f"| {row['scope']} | {row['contrast']} | {row['metric']} | {row['difference']:.8f} | [{row['low']:.8f}, {row['high']:.8f}] |")
    lines.extend(["", "These are posterior-relative measurements in a constructed example, not realized-world safety or universal necessity. "
                  "Better regret estimates alone do not establish lower reference harm. Full subgroup metrics and exact-bound diagnostics are in summary.json.", "",
                  "## Verification", "", *[f"- {name}: {value}" for name, value in summary["verification"].items()], ""])
    return "\n".join(lines)


def run_study(output_dir=None, replay=False, games_per_seed=3, data_seeds=(41101, 42101, 43101), bootstrap_samples=1000, model_dir=None):
    if games_per_seed <= 0 or bootstrap_samples <= 0:
        raise ValueError("positive evaluation and bootstrap budgets required")
    if replay and model_dir is not None:
        raise ValueError("published replay requires the bundled frozen models")
    if replay:
        published = load_json(ASSETS / "reference.json")
        data_seeds = tuple(published["test_seeds"])
        games_per_seed = published["games"] // len(data_seeds)
    torch.set_num_threads(1)
    models = load_models()
    checkpoints = {name: file_hash(ASSETS / "models" / f"{name}.pt") for name in models}
    thresholds = load_json(ASSETS / "thresholds.json")
    if model_dir is not None:
        for family in FAMILIES[1:]:
            for seed in SEEDS:
                name = f"{family}_{seed}"
                path = Path(model_dir) / f"{name}.pt"
                saved = torch.load(path, weights_only=True, map_location="cpu")
                if saved["input_protocol"] != "normalize_evidence_v1" or saved["seed"] != seed or saved["family"] != family:
                    raise ValueError("custom checkpoint identity mismatch")
                model = GraphActionModel(ModelConfig(**saved["config"]))
                model.load_state_dict(saved["state_dict"])
                models[name] = model
                checkpoints[name] = file_hash(path)
        thresholds = calibrate(models, load_json(ASSETS / "training.json")["validation"]["general"])
    output = Path(output_dir) if output_dir else Path("results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    mode = "published replay" if replay else "fresh evaluation"
    protocol = {"mode": mode, "families": list(FAMILIES), "training_seeds": list(SEEDS), "data_seeds": list(data_seeds),
                "games_per_seed": games_per_seed, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": 41101,
                "thresholds": thresholds, "threshold_source": "general validation, actual-action F1; highest threshold on ties",
                "primary": "independent minus baseline, all-case actual-action regret MAE, equal-instance weighting",
                "bounds": "oracle exact edited-optimal action set, not bounds on suboptimal learned actions",
                "checkpoint_sha256": checkpoints, "artifact_manifest_sha256": file_hash(ASSETS / "manifest.json"),
                "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write_json(output / "protocol.json", protocol)
    protocol_hash = file_hash(output / "protocol.json")
    excluded = load_json(ASSETS / "exclusions.json")
    instances = load_json(ASSETS / "test_instances.json") if replay else sample_instances(data_seeds, games_per_seed, excluded)
    observations, pairs = build_pairs(instances)
    for item in observations:
        if reference_posterior(item["view"]) != tuple(Fraction(value) for value in item["probabilities"]):
            raise AssertionError("independent exact posterior mismatch")
    print(f"{mode}: {len(instances)} instances, {len(pairs)} pairs, {len(observations)} exact-checked observations", flush=True)
    probabilities = predict_observations(observations, models, pack_observations)
    generation_seeds = {instance["game_id"]: instance["data_seed"] for instance in instances}
    rows = evaluate_pairs(observations, pairs, probabilities, thresholds, generation_seeds)
    metrics = summarize(rows, thresholds)
    intervals = paired_intervals(rows, FAMILIES, SEEDS, samples=bootstrap_samples)
    protected = [index for index, row in enumerate(next(iter(rows.values()))) if row["protected"]]
    signatures = {instance_signature(instance) for instance in instances}
    verification = {"independent_posteriors_agree": True,
                    "test_instances_unique": len(signatures) == len(instances),
                    "protected_inputs_invariant": bool(protected) and all(np.array_equal(values[pairs[index]["reference_index"]], values[pairs[index]["edited_index"]])
                        for values in probabilities.values() for index in protected),
                    "protected_incremental_loss_zero": all(group[index]["incremental_reference_regret"] == 0 for group in rows.values() for index in protected),
                    "shared_action_targets_equal": all([row["common_action_regret"] for row in rows[f"{family}_{seed}"]] ==
                        [row["common_action_regret"] for row in rows[f"baseline_{seed}"]] for family in FAMILIES for seed in SEEDS),
                    "protocol_unchanged": file_hash(output / "protocol.json") == protocol_hash}
    if not replay:
        verification["fresh_instances_disjoint"] = not signatures.intersection(excluded)
    summary = {"mode": mode, "games": len(instances), "pairs": len(pairs), "observations": len(observations),
               "protected_pairs": len(protected), "metrics": metrics, "paired_intervals": intervals, "verification": verification}
    if replay:
        summary["published_metrics_maximum_error"] = compare_reference(summary)
        verification["published_metrics_replayed"] = True
    write_json(output / "instances.json", instances)
    write_json(output / "observations.json", observations)
    write_json(output / "pairs.json", pairs)
    np.savez_compressed(output / "probabilities.npz", **probabilities)
    write_rows(output / "case_readouts.csv", [row for group in rows.values() for row in group])
    write_rows(output / "paired_intervals.csv", intervals)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(report(summary), encoding="utf-8")
    if not all(verification.values()):
        raise AssertionError(f"verification failed: {verification}")
    print(output.resolve(), flush=True)
    return output, summary


def main():
    parser = argparse.ArgumentParser(description="Evidence edits and downstream decision consequences")
    parser.add_argument("--replay", action="store_true", help="reproduce the published frozen confirmation")
    parser.add_argument("--games-per-seed", type=int, default=3)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[41101, 42101, 43101])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, help="evaluate retrained matched-control models on fresh instances")
    arguments = parser.parse_args()
    run_study(arguments.output_dir, arguments.replay, arguments.games_per_seed, tuple(arguments.data_seeds), arguments.bootstrap_samples, arguments.model_dir)


if __name__ == "__main__":
    main()