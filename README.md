# Evidence Errors and Downstream Decisions

**Which errors in evidence or relationships change a decision, and which actually worsen it?** A relation on one successful reasoning path is not necessarily required: another path, redundant evidence, or a different equally good action may preserve the decision.

This study separates evidence interpretation, action selection, and consequence estimation. Its evaluation contract accepts arbitrary aligned action-utility vectors. A Werewolf-inspired private-information voting scenario provides an inspectable example, not the definition of the research question.

## Decision Contract

For reference utilities $u$, edited-evidence utilities $v$, and the executed action $a$:

$$R_{\mathrm{reference}}(a)=\max_b u(b)-u(a),\qquad R_{\mathrm{edited}}(a)=\max_b v(b)-v(a).$$

These answer different questions. The first measures loss relative to reference information; the second measures whether the decision is justified by the edited information. Incremental reference loss subtracts the same policy's pre-edit loss and may be negative.

When edited evidence has several optimal actions, their reference losses can differ. The minimum and maximum across that exact optimal set distinguish unavoidable, possible, and tie-policy-dependent harm. These oracle bounds do not bound an arbitrary suboptimal action. A learned before/after readout must be evaluated against the action it actually selects, not an unrelated oracle tie-break.

```python
from fractions import Fraction
from decision_criticality import evaluate_decision

result = evaluate_decision(
    reference=(Fraction(3, 5), Fraction(2, 5)),
    edited=(Fraction(1, 2), Fraction(1, 2)),
    action=1,
    initial_action=0,
)
print(result["reference_regret"], result["edited_regret"])  # 0.2, 0.0
```

## Run the Study

From this directory, install Python 3.10+ dependencies:

```console
python -m pip install -e ".[notebook]"
```

Open [study.ipynb](study.ipynb) and run all cells. It illustrates the contract, displays the published results, and runs a small fresh evaluation using bundled frozen models. No external data or model download is needed after dependency installation.

The scripts expose the same workflow:

```console
python -m decision_criticality --replay
python -m decision_criticality --games-per-seed 3
python -m decision_criticality.learning --output-dir results/retrained
python -m decision_criticality --model-dir results/retrained --games-per-seed 40
```

`--replay` reproduces the published 120-instance confirmation and checks its aggregate metrics against the saved reference. Fresh evaluation uses separate seeds and rejects all 1,780 recorded prior instance signatures. The small default is an illustration, not a replacement for the published sample. Retraining uses the saved matched training slots and validation rows, then recalibrates thresholds on general validation before fresh evaluation. It does not rerun rare-evidence acquisition or claim independent training-data replication. Output directories are never overwritten.

## Example

Six players have one hidden adversary, one seer, two masons, one messenger, and one auditor. All 360 role assignments have equal prior probability. Private role facts constrain the possible worlds. Reports contain a claimed source, a target, and a binary status; source-dependent reliability and uncertain authorship require evidence integration. A forwarded copy is the same evidence, whereas an independent report is a separate signal. A trusted authorship audit identifies a source, not whether its claim is true.

Controlled edits remove evidence, retain a redundant copy, change target/status/source bindings, or operate with verified attribution and independent support/conflict. Exact fraction posteriors provide action utilities and labels. The graph model sees only the observer's visible evidence and candidate-action bindings, never hidden worlds, oracle posteriors, edit labels, or context labels. Semantic normalization deduplicates known roots and honors trusted authorship before inference.

## Findings

The final confirmation uses **120 independent instances, 28,200 paired cases, 27,155 exact-checked observations**, four model families, and three fixed training seeds. Thresholds were fixed on validation before test generation.

| Model Family | Actual-Action Regret MAE | Common-Action MAE | Edited Regret | Reference Regret |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.004045 | 0.004045 | 0.000160 | 0.048680 |
| Uniform additional training | 0.003763 | 0.003760 | 0.000114 | 0.048931 |
| Repeated near-tie evidence | 0.003762 | 0.003804 | 0.000135 | 0.048379 |
| Independent near-tie evidence | 0.003656 | 0.003671 | 0.000113 | 0.048223 |

Independent-evidence training improves actual-action regret estimation versus baseline: paired equal-instance difference **-0.000390**, 95% interval **[-0.000481, -0.000310]**. The common-action control confirms improved valuation without changing the evaluated action. Edited-evidence decisions also improve. **Lower original-reference harm is not established**, binary classification improvements are mixed, and additional independent examples do not demonstrate a near-tie decision advantage over repetition.

These are posterior-relative findings in a constructed example, not realized-world safety, universal relation necessity, or natural-play calibration. See [ANALYSIS.md](ANALYSIS.md) for methods, uncertainty, negative findings, and interpretation.

## Contents

- [decision_criticality/contracts.py](decision_criticality/contracts.py): domain-independent regret and harm contracts.
- [decision_criticality/example.py](decision_criticality/example.py): the example's observations, exact inference, edits, and encoding.
- [decision_criticality/models.py](decision_criticality/models.py): graph-action inference.
- [decision_criticality/evaluation.py](decision_criticality/evaluation.py): threshold selection and paired uncertainty.
- [decision_criticality/experiment.py](decision_criticality/experiment.py): replay and fresh evaluation.
- [decision_criticality/learning.py](decision_criticality/learning.py): matched evidence-training controls.
- [decision_criticality/assets/manifest.json](decision_criticality/assets/manifest.json): checksummed models and reproducibility inputs.

To use another domain, supply its action utilities to the contract and replace the example's observation generator, evidence edits, exact oracle, and encoder. The generic contract makes no claim that an exact oracle or sufficient representation is available in every application.

Run the tests with `python -m unittest discover -s tests -v`.