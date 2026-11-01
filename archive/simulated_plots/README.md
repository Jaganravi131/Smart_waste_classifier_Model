# Archived: scripts that fabricated the report's training figures

Everything in this folder produces **synthetic** data. None of it measured a model.

| Script | What it did |
|---|---|
| `generate_training_graphs.py` | `np.random.normal(...)` accuracy/loss curves for "50 epochs" |
| `display_training_graphs_separately.py` | same curves, separate files |
| `display_graphs_separately.py` | same curves, different layout |
| `model_training_visualization.py` | 4-panel "training" dashboard from the same formulas |
| `all_graphs_summary.py`, `separate_graphs_summary.py`, `graph_inventory.py` | textual descriptions of those fake figures |
| `verify_graphs.py`, `final_verification.py` | "verification" that only checks a PNG file exists |
| `journal_graph_descriptions.py`, `journal_submission_checklist.py` | captions and a submission checklist that present the fake figures as results |

They were moved here - not deleted - so the team can see exactly which figures in
the Phase I report were generated from random numbers:

* Fig. 6.2 "Model training performance: accuracy and loss against epoch"
* the "0.920 validation accuracy at epoch 45 / loss 0.172" sentence in §5.6 and §6.4
* Fig. 6.3 and Fig. 6.5 (confidence trends built from a sine wave plus noise)

Real replacements: `smart_bin/train_real.py` writes `runs/<name>/training_curves.png`
and `history.json` from an actual training run, and `smart_bin/evaluate_real.py`
writes the confusion matrix and per-class metrics from an actual test pass.

Do not submit any figure produced by these scripts.
