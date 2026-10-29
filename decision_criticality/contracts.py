"""Decision contracts independent of the evidence domain and learning model."""

from fractions import Fraction

import numpy as np


def evaluate_decision(reference, edited, action, initial_action=None, material=Fraction(1, 50)):
    reference = tuple(Fraction(value) for value in reference)
    edited = tuple(Fraction(value) for value in edited)
    if not reference or len(reference) != len(edited):
        raise ValueError("reference and edited utilities must be nonempty and aligned")
    if action not in range(len(reference)) or (initial_action is not None and initial_action not in range(len(reference))):
        raise ValueError("actions must index the utility vectors")
    material = Fraction(material)
    if material <= 0:
        raise ValueError("material threshold must be positive")
    optimum = max(reference)
    edited_optimum = max(edited)
    optimal_actions = tuple(index for index, value in enumerate(edited) if value == edited_optimum)
    regrets = tuple(optimum - reference[index] for index in optimal_actions)
    regret = optimum - reference[action]
    minimum, maximum = min(regrets), max(regrets)
    result = {"action": int(action), "edited_optimal_actions": list(optimal_actions),
              "reference_regret": float(regret), "exact_reference_regret": str(regret),
              "edited_regret": float(edited_optimum - edited[action]),
              "any_harm": int(regret > 0), "material_harm": int(regret >= material),
              "minimum_optimal_regret": float(minimum), "maximum_optimal_regret": float(maximum),
              "possible_any_harm": int(maximum > 0), "unavoidable_any_harm": int(minimum > 0),
              "possible_material_harm": int(maximum >= material), "unavoidable_material_harm": int(minimum >= material),
              "policy_dependent": minimum == 0 < maximum, "edited_tie": len(optimal_actions) > 1}
    if initial_action is not None:
        result.update({"initial_action": int(initial_action), "action_changed": initial_action != action,
                       "initial_reference_regret": float(optimum - reference[initial_action]),
                       "incremental_reference_regret": float(reference[initial_action] - reference[action])})
    return result


def consequence_readout(reference_prediction, edited_prediction):
    before, after = np.asarray(reference_prediction, dtype=float), np.asarray(edited_prediction, dtype=float)
    if before.ndim != 1 or not before.size or after.shape != before.shape:
        raise ValueError("predictions must be nonempty aligned vectors")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("predictions must be finite")
    action, initial_action = int(after.argmax()), int(before.argmax())
    return {"action": action, "initial_action": initial_action,
            "predicted_regret": float(before.max() - before[action])}