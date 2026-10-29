# Evidence Edits and Downstream Decisions

Mode: fresh evaluation. Instances: 9; paired cases: 2115; exact-checked observations: 2043.

Targets evaluate each model's executed action. Common-action MAE holds the baseline's action fixed. Reference loss and edited-evidence suboptimality are distinct. Bounds range over oracle edited-optimal actions only.

| Family | Regret MAE | Common-action MAE | Edited regret | Reference regret | Any F1 | Material F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00354501 | 0.00354501 | 0.00013625 | 0.05363422 | 0.98249237 | 0.98623889 |
| uniform | 0.00328364 | 0.00336012 | 0.00009781 | 0.05465940 | 0.98201440 | 0.98688743 |
| repeated | 0.00320872 | 0.00327037 | 0.00009670 | 0.05279610 | 0.98079293 | 0.98500952 |
| independent | 0.00318513 | 0.00325904 | 0.00009105 | 0.05263630 | 0.98248744 | 0.98778788 |

## Paired Equal-Instance Contrasts

Differences average fixed training seeds inside whole instances. Intervals resample whole instances within generation-seed strata, conditional on models and thresholds. Secondary comparisons are unadjusted. Case-weighted means above can differ from equal-instance contrasts.

| Scope | Contrast | Metric | Difference | 95% interval |
| --- | --- | --- | ---: | --- |
| all | independent_minus_baseline | regret_error | -0.00035987 | [-0.00070695, -0.00005750] |
| all | independent_minus_baseline | common_action_error | -0.00028596 | [-0.00068651, 0.00006679] |
| all | independent_minus_baseline | edited_regret | -0.00004520 | [-0.00007193, -0.00001846] |
| all | independent_minus_baseline | reference_regret | -0.00099792 | [-0.00379050, 0.00247280] |
| all | independent_minus_repeated | regret_error | -0.00002359 | [-0.00019341, 0.00015133] |
| all | independent_minus_repeated | common_action_error | -0.00001133 | [-0.00017188, 0.00015811] |
| all | independent_minus_repeated | edited_regret | -0.00000565 | [-0.00004814, 0.00003118] |
| all | independent_minus_repeated | reference_regret | -0.00015980 | [-0.00088179, 0.00054038] |
| near_tie | independent_minus_baseline | regret_error | 0.00020320 | [-0.00102478, 0.00210311] |
| near_tie | independent_minus_baseline | common_action_error | 0.00012321 | [-0.00047979, 0.00113734] |
| near_tie | independent_minus_baseline | edited_regret | -0.00032915 | [-0.00052756, 0.00000000] |
| near_tie | independent_minus_baseline | reference_regret | -0.00423515 | [-0.02163976, 0.00893430] |
| near_tie | independent_minus_repeated | regret_error | 0.00074770 | [0.00011936, 0.00117559] |
| near_tie | independent_minus_repeated | common_action_error | 0.00073985 | [-0.00004694, 0.00125822] |
| near_tie | independent_minus_repeated | edited_regret | 0.00000000 | [0.00000000, 0.00000000] |
| near_tie | independent_minus_repeated | reference_regret | 0.00000000 | [0.00000000, 0.00000000] |

These are posterior-relative measurements in a constructed example, not realized-world safety or universal necessity. Better regret estimates alone do not establish lower reference harm. Full subgroup metrics and exact-bound diagnostics are in summary.json.

## Verification

- independent_posteriors_agree: True
- test_instances_unique: True
- protected_inputs_invariant: True
- protected_incremental_loss_zero: True
- shared_action_targets_equal: True
- protocol_unchanged: True
- fresh_instances_disjoint: True
