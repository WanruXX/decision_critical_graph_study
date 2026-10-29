# Analysis

## Research Question

An evidence or mapping error matters through its downstream consequences, not merely because it changes a representation. Evidence can be redundant; different actions can be equally good; the same edit can be harmless in one information state and damaging in another. The study therefore evaluates consequences for a stated decision contract, rather than assigning universal importance to individual relations.

The private-information voting example makes evidence provenance, source reliability, private constraints, and alternative actions explicit. Its conclusions are conditional on that example's information model. The utility-vector contract is domain-independent; the generator, graph encoding, and learned posterior are not universal algorithms for necessity detection.

## Example and Evidence Model

There are six players with roles wolf, seer, mason, mason, messenger, and auditor. The 360 assignments are equiprobable. Each non-wolf observer knows its own role. The seer privately inspects another uniformly sampled player; the masons know each other; the messenger receives a third report root and a forwarded copy; the auditor verifies the true author of one public report.

Each report's actual author and target are independently uniform over six players. Claimed authorship is correct with probability 0.8, otherwise uniformly wrong. Report accuracy is 0.9 for a seer, 0.1 for a wolf, and 0.5 for other roles. Independent roots are independent conditional on roles. Copies count once. Trusted private inspections and audits are distinct from these noisy reports. Authorship is not truth, and reports are not strategic dialogue.

The exact posterior marginalizes unknown authors and role assignments with fraction arithmetic. A separate world-by-world implementation checks it. Intervened and constructed evidence is interpreted under the original signal-channel model. Consequently, posterior-relative regret is not guaranteed calibration to actual outcomes or to the context-selection mechanism.

The model encodes visible players, private facts, reports, source/target bindings, copied origins, and verified authorship. A 40-feature node encoder feeds six directed message-passing layers of width 32, residual SiLU updates, and an action-scoring head. The normalized input removes copies of known roots and replaces claimed authors with verified authors where available, while retaining independent roots. This normalization is engineered semantic knowledge, not learned provenance recovery. Hidden roles and oracle values never enter model features.

## What Counts as Harm

Reference regret is optimal utility under original evidence minus the utility of the executed edited action. Edited regret instead evaluates that action under edited evidence. These can move differently: a decision can be correct for corrupted evidence and still lose reference value. Signed incremental reference regret subtracts the same model's pre-edit loss; it is zero for an unchanged action even when the model was already wrong.

Exact edited-optimal actions define minimum and maximum possible reference regret. A positive minimum means every optimal edited decision is harmful; a positive maximum means at least one is harmful. A zero minimum with positive maximum identifies tie-policy-dependent harm. These bounds describe only the exact edited-optimal set, not suboptimal learned actions. Exact material harm is regret >= 1/50. A unique top-two gap <= 1/50 defines a descriptive near-tie subset, not an inference tolerance or a bound on losing to a third candidate.

The learned readout values the model's actual edited argmax using its learned original posterior. Its target is exact reference regret for that same action. Comparing it to a fixed oracle action can manufacture apparent valuation failure when the learner merely chooses another exact optimum. Conversely, scoring its actual action does not excuse a genuinely suboptimal edited vote. A common-action control holds the baseline's edited action fixed across all model families, separately for each training seed, to isolate valuation changes.

## Matched Learning Controls

Three evidence-training conditions warm-start from each normalized baseline at seeds 42, 73, and 101: uniform original evidence, repeated near-tie evidence, and new independent near-tie evidence. All use soft-target cross entropy, AdamW learning rate 0.0003, weight decay 0.0001, clipping at 5, batch size 64, and ten epochs. Each fit executes 940 optimizer steps and 60,000 presentations. The enriched arms have identical 4,800-row background slots and 1,200 near-tie slots per epoch, matched by role, condition, and exact margin bin (0, 0.005], (0.005, 0.01], or (0.01, 0.02].

Repetition uses 61 distinct normalized inputs from 47 instances, contained in the original 63 near-tie rows. The independent arm uses 372 new inputs from 372 independent instances, at most one acquired row per instance. The view and condition are sampled before label-based rejection; learned error does not select examples. Acquisition accepts 550 distinct inputs in 100,000 attempts and partitions them into 372 training and 178 targeted validation examples. Scarce strata repeat fewer novel inputs to preserve matched exposure. This matches fitting budgets, not acquisition cost, and does not prove that possible novel inputs are exhausted.

Validation contains 3,178 rows: all 25 observation variants from 120 general instances plus one selected row from each of 178 targeted instances, including 204 near-ties. Each fit selects by validation near regret, then overall regret and cross entropy, with epoch zero eligible, cross-entropy drift <= 0.01 and posterior MAE <= 0.03. Effects include checkpoint selection. The shared validation procedure selects the independent-evidence family before final evaluation. The targeted supplement is not an unconditioned deployment sample.

Bundled training slots and validation rows reproduce these controlled fits from the supplied starting models. They do not regenerate evidence acquisition or constitute a new acquisition replication. Starting models were trained on 240 instances and selected on 60 validation instances with soft-target cross entropy, AdamW 0.003, batch size 64, at most 45 epochs, patience 8, and epoch zero eligible. Frozen models and portable inputs are checksummed in the artifact manifest.

## Final Confirmation

All twelve models are frozen. Actual-action any/material-harm thresholds maximize F1 on the existing 120 general validation instances, breaking ties toward the highest threshold. These instances also supported checkpoint selection, so this is not independently held-out calibration. Protocol and thresholds precede fresh test generation; no test tuning or favorable-seed selection occurs.

The confirmation uses 120 new instances, 40 from each generation seed 38101/39101/40101. Whole instances retain all five perspectives and paired variants. There are 28,200 paired cases, 27,155 independently checked observations, 489 unique-optimum near-ties across 90 instances, and 3,376 policy-dependent cases across 120 instances. Full original-instance signatures exclude every preceding training, validation, acquisition, and test collection. Repeated structural patterns remain possible.

The README table gives case-weighted means across fixed training seeds. The following differences instead average seeds within each instance and weight instances equally. Intervals use 1,000 whole-instance bootstrap resamples within generation-seed strata, conditional on models, acquisition, and thresholds. Training seeds are not additional independent test instances. Secondary intervals are unadjusted; these intervals do not capture repeated-training or repeated-acquisition uncertainty.

| Independent Minus Baseline, All Cases | Difference | 95% Interval |
| --- | ---: | ---: |
| Actual-action regret MAE, primary | -0.000390 | [-0.000481, -0.000310] |
| Common-action valuation MAE | -0.000374 | [-0.000470, -0.000289] |
| Edited regret | -0.0000470 | [-0.0000712, -0.0000233] |
| Original-reference regret | -0.000456 | [-0.001067, 0.000125] |
| Signed incremental reference regret | -0.000419 | [-0.001026, 0.000166] |
| Any-harm classification error | +0.000378 | [-0.001478, 0.002045] |
| Material-harm classification error | -0.001738 | [-0.003558, -0.000236] |

The primary result supports improved consequence valuation. The common-action improvement shows this is not merely an artifact of choosing different actions. Binary results are mixed: baseline versus independent any-harm F1 is 0.9695 versus 0.9685; material-harm F1 is 0.9781 versus 0.9809. Targets differ with executed actions, so these are not common-target classifier comparisons. Better estimates and lower edited regret do not establish lower original-reference harm.

Versus repetition, independent evidence improves common-action MAE by 0.000133 [0.000089, 0.000183] in magnitude and all-case edited regret by 0.0000215 [0.0000108, 0.0000342]. Its actual-action MAE difference -0.000107 [-0.000190, 0.000010] and reference-regret difference -0.000156 [-0.000702, 0.000497] remain inconclusive.

Both enriched arms reduce near-tie edited regret versus uniform training: independent minus uniform -0.000279 [-0.000444, -0.000129], repetition minus uniform -0.000418 [-0.000652, -0.000231]. Independent minus repetition is +0.000139 [-0.000086, 0.000394]. Thus new evidence does not demonstrate an additional near-tie decision gain over repetition. Equal-instance and case-weighted subset summaries can differ in sign.

All 1,920 protected copy/audit cases preserve model probabilities and actions exactly and have zero incremental reference loss. This does not assert zero absolute loss for an already-suboptimal model. Oracle possible/unavoidable bounds remain diagnostics, not learned bound detectors.

## Interpretation

The supported conclusion is specific: evidence semantics and the decision contract determine whether a relation error matters; learned posterior readouts can estimate the consequences for their executed actions; improved inference can improve this valuation. Probability fidelity, action ranking, and reference harm remain different objectives.

This study does not establish universal relation necessity, automatic recovery of deleted facts, full-game strategy, realized-world safety, natural-play calibration, or generalization to other tasks. Exact solvers are controls and teachers, not learned reasoning discoveries. A finite set without input-label conflicts is not proof of universal identifiability. The small notebook run demonstrates the workflow; only the full frozen replay reproduces the reported confirmation.