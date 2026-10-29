# Evidence Edits and Downstream Decisions

Mode: published replay. Instances: 120; paired cases: 28200; exact-checked observations: 27155.

Targets evaluate each model's executed action. Common-action MAE holds the baseline's action fixed. Reference loss and edited-evidence suboptimality are distinct. Bounds range over oracle edited-optimal actions only.

| Family | Regret MAE | Common-action MAE | Edited regret | Reference regret | Any F1 | Material F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00404505 | 0.00404505 | 0.00016018 | 0.04867956 | 0.96945857 | 0.97809588 |
| uniform | 0.00376280 | 0.00375952 | 0.00011350 | 0.04893060 | 0.96979352 | 0.98118077 |
| repeated | 0.00376232 | 0.00380376 | 0.00013467 | 0.04837917 | 0.96775857 | 0.97639180 |
| independent | 0.00365551 | 0.00367124 | 0.00011321 | 0.04822312 | 0.96853024 | 0.98093769 |

## Paired Equal-Instance Contrasts

Differences average fixed training seeds inside whole instances. Intervals resample whole instances within generation-seed strata, conditional on models and thresholds. Secondary comparisons are unadjusted. Case-weighted means above can differ from equal-instance contrasts.

| Scope | Contrast | Metric | Difference | 95% interval |
| --- | --- | --- | ---: | --- |
| all | independent_minus_baseline | regret_error | -0.00038954 | [-0.00048126, -0.00031033] |
| all | independent_minus_baseline | common_action_error | -0.00037380 | [-0.00046972, -0.00028881] |
| all | independent_minus_baseline | edited_regret | -0.00004697 | [-0.00007120, -0.00002327] |
| all | independent_minus_baseline | reference_regret | -0.00045643 | [-0.00106740, 0.00012529] |
| all | independent_minus_repeated | regret_error | -0.00010682 | [-0.00018979, 0.00001018] |
| all | independent_minus_repeated | common_action_error | -0.00013252 | [-0.00018299, -0.00008943] |
| all | independent_minus_repeated | edited_regret | -0.00002146 | [-0.00003417, -0.00001079] |
| all | independent_minus_repeated | reference_regret | -0.00015605 | [-0.00070161, 0.00049650] |
| near_tie | independent_minus_baseline | regret_error | -0.00071447 | [-0.00197552, 0.00043889] |
| near_tie | independent_minus_baseline | common_action_error | -0.00043061 | [-0.00082505, -0.00004327] |
| near_tie | independent_minus_baseline | edited_regret | -0.00046831 | [-0.00075283, -0.00019953] |
| near_tie | independent_minus_baseline | reference_regret | -0.00076759 | [-0.00664445, 0.00478310] |
| near_tie | independent_minus_repeated | regret_error | 0.00052745 | [-0.00027366, 0.00152542] |
| near_tie | independent_minus_repeated | common_action_error | -0.00020307 | [-0.00050979, 0.00008222] |
| near_tie | independent_minus_repeated | edited_regret | 0.00013881 | [-0.00008573, 0.00039401] |
| near_tie | independent_minus_repeated | reference_regret | 0.00466486 | [-0.00026435, 0.01004261] |

These are posterior-relative measurements in a constructed example, not realized-world safety or universal necessity. Better regret estimates alone do not establish lower reference harm. Full subgroup metrics and exact-bound diagnostics are in summary.json.

## Verification

- independent_posteriors_agree: True
- test_instances_unique: True
- protected_inputs_invariant: True
- protected_incremental_loss_zero: True
- shared_action_targets_equal: True
- protocol_unchanged: True
- published_metrics_replayed: True
