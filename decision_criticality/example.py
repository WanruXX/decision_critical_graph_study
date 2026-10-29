"""A private-information voting example of evidence-dependent decisions."""

from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import permutations
import json

import numpy as np
import torch


ROLES = ("wolf", "seer", "mason", "messenger", "auditor")
ASSIGNMENTS = tuple(sorted(set(permutations((0, 1, 2, 2, 3, 4)))))
WORLDS = np.asarray(ASSIGNMENTS, dtype=int)
CONTEXTS = ("original", "single_origin", "forwarded_copy", "independent_support", "independent_conflict", "unverified_source")
EDITS = ("remove_origin", "remove_one_copy", "changed_target", "changed_status", "changed_author")
CONDITIONS = ("clean", "missing_report", "changed_target", "changed_status", "changed_author")


def distinct_reports(view):
    roots = {}
    for report in view["reports"]:
        content = (report["author"], report["target"], report["wolf_claim"])
        if report["root"] in roots and roots[report["root"]] != content:
            raise ValueError("copies must preserve content and attribution")
        roots[report["root"]] = content
    return roots


def posterior(view):
    mask = WORLDS[:, view["observer"]] == view["own_role"]
    if view.get("inspection") is not None:
        target, status = view["inspection"]
        mask &= (WORLDS[:, target] == 0) == status
    if view.get("peer") is not None:
        mask &= WORLDS[:, view["peer"]] == 2
    weights = [int(value) for value in mask]
    for root, (claimed_author, target, claim) in distinct_reports(view).items():
        verified = view["audit"].get(root)
        numerator = np.zeros(len(WORLDS), dtype=int)
        for author in range(6):
            if verified is not None and author != verified:
                continue
            source_weight = 1 if verified is not None else (20 if author == claimed_author else 1)
            truth = WORLDS[:, target] == 0
            accuracy = np.where(WORLDS[:, author] == 1, 9, np.where(WORLDS[:, author] == 0, 1, 5))
            numerator += source_weight * np.where(truth == bool(claim), accuracy, 10 - accuracy)
        weights = [weight * int(likelihood) for weight, likelihood in zip(weights, numerator)]
    denominator = sum(weights)
    if not denominator:
        raise ValueError("private evidence has no compatible world")
    return tuple(Fraction(sum(weight for weight, world in zip(weights, ASSIGNMENTS) if world[target] == 0), denominator)
                 for target in range(6))


def reference_posterior(view):
    totals = [Fraction(0) for _ in range(6)]
    for world in ASSIGNMENTS:
        if world[view["observer"]] != view["own_role"]:
            continue
        if view.get("peer") is not None and world[view["peer"]] != 2:
            continue
        if view.get("inspection") is not None:
            target, status = view["inspection"]
            if (world[target] == 0) != bool(status):
                continue
        likelihood = Fraction(1)
        for root, (claimed_author, target, claim) in distinct_reports(view).items():
            verified = view["audit"].get(root)
            report_likelihood = Fraction(0)
            for author in range(6):
                if verified is not None and author != verified:
                    continue
                source_probability = Fraction(1) if verified is not None else Fraction(20 if author == claimed_author else 1, 25)
                reliability = Fraction(9, 10) if world[author] == 1 else (Fraction(1, 10) if world[author] == 0 else Fraction(1, 2))
                report_likelihood += source_probability * (reliability if (world[target] == 0) == bool(claim) else 1 - reliability)
            likelihood *= report_likelihood
        totals[world.index(0)] += likelihood
    denominator = sum(totals)
    if not denominator:
        raise ValueError("private evidence has no compatible world")
    return tuple(value / denominator for value in totals)


def generate_instance(random, instance_id):
    world = ASSIGNMENTS[int(random.integers(len(ASSIGNMENTS)))]
    roots, true_authors = [], {}
    for index in range(3):
        author = int(random.integers(6))
        claimed_author = author if random.random() < 0.8 else int(random.choice([player for player in range(6) if player != author]))
        target = int(random.integers(6))
        reliability = 0.9 if world[author] == 1 else (0.1 if world[author] == 0 else 0.5)
        truth = world[target] == 0
        claim = truth if random.random() < reliability else not truth
        root = f"r{index}"
        roots.append({"root": root, "author": claimed_author, "target": target, "wolf_claim": bool(claim), "relay": False})
        true_authors[root] = author
    views = []
    for observer, role in enumerate(world):
        if role == 0:
            continue
        reports = [dict(report) for report in roots[:2]]
        view = {"observer": observer, "own_role": role, "inspection": None, "peer": None, "audit": {}, "reports": reports}
        if role == 1:
            target = int(random.choice([player for player in range(6) if player != observer]))
            view["inspection"] = (target, world[target] == 0)
        elif role == 2:
            view["peer"] = next(player for player, other_role in enumerate(world) if other_role == 2 and player != observer)
        elif role == 3:
            reports.extend((dict(roots[2]), {**roots[0], "relay": True}))
        elif role == 4:
            root = roots[int(random.integers(2))]["root"]
            view["audit"][root] = true_authors[root]
        views.append(view)
    return {"game_id": instance_id, "roles": world, "true_wolf": world.index(0), "true_authors": true_authors, "views": views}


def instance_signature(instance):
    content = {key: value for key, value in instance.items() if key not in ("game_id", "data_seed")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def sample_instances(seeds, count, excluded=()):
    if not seeds or len(set(seeds)) != len(seeds) or count <= 0:
        raise ValueError("distinct seeds and a positive per-seed count are required")
    seen, instances = set(excluded), []
    for seed in seeds:
        random = np.random.default_rng(seed)
        accepted = 0
        while accepted < count:
            instance = generate_instance(random, f"comparison_{seed}_{accepted:05d}")
            signature = instance_signature(instance)
            if signature in seen:
                continue
            seen.add(signature)
            instance["data_seed"] = int(seed)
            instances.append(instance)
            accepted += 1
    return instances


def normalize_evidence(view):
    result = deepcopy(view)
    result["reports"] = [{"root": root, "author": view["audit"].get(root, author), "target": target,
                          "wolf_claim": bool(claim), "relay": False}
                         for root, (author, target, claim) in distinct_reports(view).items()]
    result["reports"].sort(key=lambda report: (report["author"], report["target"], report["wolf_claim"], report["root"] in view["audit"]))
    return result


def encode_action(view, action):
    order = [view["observer"]] + [player for player in range(6) if player != view["observer"]]
    local = {player: index for index, player in enumerate(order)}
    nodes, adjacency = torch.zeros(10, 40), torch.zeros(10, 10)
    for player in range(6):
        index = local[player]
        nodes[index, 0] = nodes[index, 12 + index] = 1
    nodes[0, 3] = nodes[0, 5 + view["own_role"]] = nodes[0, 10] = nodes[local[action], 4] = 1
    if view.get("peer") is not None:
        index = local[view["peer"]]
        nodes[index, 10] = adjacency[0, index] = 1
    if view.get("inspection") is not None:
        player, status = view["inspection"]
        index = local[player]
        nodes[index, 11 if status else 10] = adjacency[0, index] = 1
    roots = {}
    for offset, report in enumerate(view["reports"]):
        index = 6 + offset
        nodes[index, 1] = nodes[index, 18 if report["wolf_claim"] else 19] = 1
        nodes[index, 20] = report["relay"]
        author, target = local[report["author"]], local[report["target"]]
        nodes[index, 21 + author] = nodes[index, 27 + target] = 1
        adjacency[author, index] = adjacency[index, target] = 1
        if report["root"] in roots:
            adjacency[roots[report["root"]], index] = 1
        else:
            roots[report["root"]] = index
        if report["root"] in view["audit"]:
            verified = local[view["audit"][report["root"]]]
            nodes[index, 33] = nodes[index, 34 + verified] = adjacency[verified, index] = 1
    return nodes, adjacency, local[action]


def pack_observations(views):
    encodings = [[encode_action(normalize_evidence(view), player) for player in range(6) if player != view["observer"]]
                 for view in views]
    if not encodings:
        raise ValueError("at least one observation is required")
    return (torch.stack([torch.stack([item[0] for item in row]) for row in encodings]),
            torch.stack([torch.stack([item[1] for item in row]) for row in encodings]),
            torch.tensor([[item[2] for item in row] for row in encodings]))


def contextualize(view, context, root):
    if context not in CONTEXTS:
        raise ValueError("unknown evidence context")
    result = deepcopy(view)
    if context == "original":
        return result
    reports, seen = [], set()
    for report in result["reports"]:
        if report["root"] not in seen:
            reports.append({**report, "relay": False})
            seen.add(report["root"])
    result["reports"] = reports
    selected = next(report for report in reports if report["root"] == root)
    if context == "unverified_source":
        result["audit"].pop(root, None)
    elif context != "single_origin":
        extra = {**selected, "relay": context == "forwarded_copy"}
        if context != "forwarded_copy":
            extra["root"] = f"independent_{root}"
        if context == "independent_conflict":
            extra["wolf_claim"] = not extra["wolf_claim"]
        result["reports"].append(extra)
    return result


def edit_evidence(view, root, edit):
    if edit not in EDITS:
        raise ValueError("unknown evidence edit")
    result = deepcopy(view)
    if edit == "remove_origin":
        result["reports"] = [report for report in result["reports"] if report["root"] != root]
    elif edit == "remove_one_copy":
        index = next(index for index, report in enumerate(result["reports"]) if report["root"] == root)
        del result["reports"][index]
    else:
        for report in result["reports"]:
            if report["root"] == root:
                if edit == "changed_status":
                    report["wolf_claim"] = not report["wolf_claim"]
                else:
                    field = "target" if edit == "changed_target" else "author"
                    report[field] = (report[field] + 1) % 6
    return result


def observation_variant(view, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown observation condition")
    if condition == "clean":
        return deepcopy(view)
    edit = "remove_origin" if condition == "missing_report" else condition
    return edit_evidence(view, view["reports"][0]["root"], edit)


def build_pairs(instances):
    observations, indices, pairs = [], {}, []

    def intern(view):
        key = json.dumps(view, sort_keys=True)
        if key not in indices:
            indices[key] = len(observations)
            observations.append({"view": view, "probabilities": [str(value) for value in posterior(view)]})
        return indices[key]

    for instance in instances:
        for view in instance["views"]:
            for root in distinct_reports(view):
                for context in CONTEXTS:
                    if context == "unverified_source" and root not in view["audit"]:
                        continue
                    before = contextualize(view, context, root)
                    for edit in EDITS:
                        if edit == "remove_one_copy" and context != "forwarded_copy":
                            continue
                        pairs.append({"game_id": instance["game_id"], "observer": view["observer"], "role": ROLES[view["own_role"]],
                                      "root": root, "context": context, "edit": edit,
                                      "reference_index": intern(before), "edited_index": intern(edit_evidence(before, root, edit))})
    return observations, pairs