# Evidence Edits and Downstream Decisions

Mode: fresh evaluation. Instances: 1; paired cases: 235; exact-checked observations: 227.

Targets evaluate each model's executed action. Common-action MAE holds the baseline's action fixed. Reference loss and edited-evidence suboptimality are distinct. Bounds range over oracle edited-optimal actions only.

| Family | Regret MAE | Common-action MAE | Edited regret | Reference regret | Any F1 | Material F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00559617 | 0.00559617 | 0.00001630 | 0.07074146 | 0.97969238 | 0.98632155 |
| uniform | 0.00503301 | 0.00494469 | 0.00002812 | 0.07241766 | 0.97307494 | 0.98659120 |
| repeated | 0.00484738 | 0.00486031 | 0.00002812 | 0.06990969 | 0.97941733 | 0.97852872 |
| independent | 0.00404695 | 0.00395655 | 0.00003180 | 0.07288235 | 0.97316227 | 0.98637185 |

## Paired Equal-Instance Contrasts

Differences average fixed training seeds inside whole instances. Intervals resample whole instances within generation-seed strata, conditional on models and thresholds. Secondary comparisons are unadjusted. Case-weighted means above can differ from equal-instance contrasts.

| Scope | Contrast | Metric | Difference | 95% interval |
| --- | --- | --- | ---: | --- |
| all | independent_minus_baseline | regret_error | -0.00154922 | [-0.00154922, -0.00154922] |
| all | independent_minus_baseline | common_action_error | -0.00163962 | [-0.00163962, -0.00163962] |
| all | independent_minus_baseline | edited_regret | 0.00001551 | [0.00001551, 0.00001551] |
| all | independent_minus_baseline | reference_regret | 0.00214089 | [0.00214089, 0.00214089] |
| all | independent_minus_repeated | regret_error | -0.00080043 | [-0.00080043, -0.00080043] |
| all | independent_minus_repeated | common_action_error | -0.00090376 | [-0.00090376, -0.00090376] |
| all | independent_minus_repeated | edited_regret | 0.00000368 | [0.00000368, 0.00000368] |
| all | independent_minus_repeated | reference_regret | 0.00297266 | [0.00297266, 0.00297266] |
| near_tie | independent_minus_baseline | regret_error | 0.00328764 | [0.00328764, 0.00328764] |
| near_tie | independent_minus_baseline | common_action_error | -0.00088389 | [-0.00088389, -0.00088389] |
| near_tie | independent_minus_baseline | edited_regret | 0.00052054 | [0.00052054, 0.00052054] |
| near_tie | independent_minus_baseline | reference_regret | 0.06176958 | [0.06176958, 0.06176958] |
| near_tie | independent_minus_repeated | regret_error | -0.00117758 | [-0.00117758, -0.00117758] |
| near_tie | independent_minus_repeated | common_action_error | -0.00053543 | [-0.00053543, -0.00053543] |
| near_tie | independent_minus_repeated | edited_regret | 0.00012350 | [0.00012350, 0.00012350] |
| near_tie | independent_minus_repeated | reference_regret | 0.01563865 | [0.01563865, 0.01563865] |

These are posterior-relative measurements in a constructed example, not realized-world safety or universal necessity. Better regret estimates alone do not establish lower reference harm. Full subgroup metrics and exact-bound diagnostics are in summary.json.

## Verification

- independent_posteriors_agree: True
- test_instances_unique: True
- protected_inputs_invariant: True
- protected_incremental_loss_zero: True
- shared_action_targets_equal: True
- protocol_unchanged: True
- fresh_instances_disjoint: True
