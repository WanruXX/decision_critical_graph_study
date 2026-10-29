"""Validation-only thresholds and paired whole-instance comparisons."""

import math

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score


def select_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or not labels.size or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("aligned nonempty binary labels and scores required")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    if not labels.any():
        return math.nextafter(float(max(1, scores.max())), math.inf)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(max(thresholds[np.isclose(values, values.max(), rtol=0, atol=1e-12)]))


def classification_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= threshold
    return {"positives": int(labels.sum()), "prevalence": float(labels.mean()),
            "accuracy": float(accuracy_score(labels, predictions)), "f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "average_precision": float(average_precision_score(labels, scores)) if labels.any() else None,
            "threshold": float(threshold)}


def paired_intervals(rows, families, seeds, samples=1000, random_seed=41101):
    if samples <= 0:
        raise ValueError("positive bootstrap sample count required")
    fields = ("regret_error", "common_action_error", "edited_regret", "reference_regret",
              "incremental_reference_regret", "any_harm_error", "material_harm_error", "suboptimal_action")
    first = rows[f"{families[0]}_{seeds[0]}"]
    identity = lambda row: tuple(row[field] for field in ("game_id", "observer", "root", "context", "edit"))
    expected = [identity(row) for row in first]
    if any([identity(row) for row in group] != expected for group in rows.values()):
        raise ValueError("paired comparisons require identical case ordering")
    random, result = np.random.default_rng(random_seed), []
    for scope in ("all", "near_tie", "edited_tie", "policy_dependent"):
        blocks = {}
        for index, row in enumerate(first):
            if scope == "all" or row[scope]:
                blocks.setdefault(row["game_id"], []).append(index)
        instances = sorted(blocks)
        if not instances:
            continue
        strata = {}
        for index, instance in enumerate(instances):
            strata.setdefault(first[blocks[instance][0]]["data_seed"], []).append(index)
        draws = [random.choice(indices, (samples, len(indices)), replace=True) for indices in strata.values()]
        means = {}
        for family in families:
            values = np.mean([[[row[field] for field in fields] for row in rows[f"{family}_{seed}"]]
                              for seed in seeds], axis=0)
            means[family] = np.asarray([values[blocks[instance]].mean(0) for instance in instances])
        contrasts = [(family, families[0]) for family in families[1:]]
        contrasts += [("independent", "repeated"), ("repeated", "uniform"), ("independent", "uniform")]
        for treatment, control in contrasts:
            if treatment not in means or control not in means:
                continue
            differences = means[treatment] - means[control]
            bootstrap = sum(differences[draw].sum(1) for draw in draws) / len(instances)
            for column, field in enumerate(fields):
                result.append({"scope": scope, "contrast": f"{treatment}_minus_{control}", "metric": field,
                               "games": len(instances), "difference": float(differences[:, column].mean()),
                               "low": float(np.quantile(bootstrap[:, column], 0.025)),
                               "high": float(np.quantile(bootstrap[:, column], 0.975))})
    return result