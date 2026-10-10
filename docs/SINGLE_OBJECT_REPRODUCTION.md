# Single-object reproduction

Run commands from the repository root. Python 3.11 and the root
`requirements.txt` describe the submission environment. The scientific data
are external; obtain the approved original dataset from the authors.
Full training is a GPU workload, not part of the data-free checks below.
Do not run the entire `configs/` tree: it includes historical experiments and
the separate pair study. Choose a specific config or family.

## 1. Check the included evidence (no data, Torch, or GPU needed)

```text
python tools/single_object_artifacts.py verify
python tools/single_object_summary.py
```

These verify the archived files and recompute the five main single-object
benchmark rows from saved per-seed measurements. Output:
`results/generated/single_object/summary.csv`. This is numerical reconstruction
of existing measurements, not new inference or proof of retraining equivalence.
The package includes 586 primary metric files and their run configs, plus all
78 pretrained checkpoints required by the 114 original FT configurations.
See [evidence and plotting instructions](../results/single_object/README.md).

## 2. Data and evaluation conventions

Pass the directory containing this exact structure to `--data_dir`:

```text
DATA_ROOT/
  Signal/signal_combined.csv
  Signal/signal_combined.npz
  Background/bkg_combined.csv
  Background/bkg_combined.npz
```

CSV rows and NPZ objects must refer to the same `(sample, eventNumber,
tob_index)` entries. Keep source energies in MeV. NPZ arrays include
`event_nums`, `X_tensors`, `X_em2_tensors`, and `X_feats`; do not substitute a
different population or grid without treating it as a new experiment.
Original data hashes are recorded in `results/single_object/test_counts.json`.

The code splits complete events 70/10/20, keeps signal/background event IDs
distinct, and fits preprocessing on the training split. Each seed has its own
split. The single-object **published primary comparison recalibrates thresholds
on test background**, for both the model and TOB baseline, at requested 0.5%
event FPR. Efficiency counts truth-matched tau objects, not signal events.
Validation-calibrated test performance is a separate diagnostic. Do not equate
this protocol or metric with the pair study's frozen-threshold event acceptance.

## 3. Which configuration reproduces which main row?

Replace `SEED` with **42, 123, and 456**, running each separately. These are
independent models, not an ensemble. The exact JSON paths, archived run names,
and parent paths are also in `results/single_object/benchmark_map.json`.

| Paper row | Configuration | Global efficiency (mean +/- SE, %) |
| --- | --- | --- |
| E2+V4+F, BCE, OR | `configs/cnn_v13_or_bce_sSEED/v13_or_em2ladder_dual_sSEED.json` | 67.97 +/- 0.55 |
| E1+V4+F, BCE+FT | `configs/cnn_v12_ft_em2single_dual_fracbce/v12_ft_em2single_dual_fracbce_sSEED.json` | 69.61 +/- 0.29 |
| E3+V2+F, FT | `configs/cnn_v12_ft_em2stride_conv2x2_frac/v12_ft_em2stride_conv2x2_frac_sSEED.json` | 69.08 +/- 0.35 |
| MLP, CT+F, IF | `configs/nn_only_core_fraction_sSEED/c002_sSEED.json` | 68.40 +/- 0.21 |
| V3+F, IF, OR | `configs/cnn_v8_or_serial_invfreq/v8_or_serial_invfreq_sSEED.json` | 67.47 +/- 0.22 |

CT is the 45-cell tensor; F is `em2_best_3x3_fraction`; E1--E3 process fine
EM2 and V1--V4 process CT. IF means **inverse-frequency** weighting, not the
distinct power-law `p=-1` weighting. Both redistribute signal weights only.
The JSON config, including `loss.weighting.profile`, is authoritative.

## 4. Train and evaluate an ordinary model

After installing dependencies in the chosen GPU environment, a complete
seed-42 example is (replace `DATA_ROOT` with your actual data directory):

```text
python src/train.py --config configs/nn_only_core_fraction_s42/c002_s42.json --data_dir "DATA_ROOT" --experiments_dir experiments/reproduced_mlp_if
python src/evaluate.py --experiments_dir experiments/reproduced_mlp_if --recalc
```

Repeat training with the corresponding seed-123 and seed-456 configs into the
same output directory, then evaluate it again to produce three-seed averages.
Do not mix model families in that output directory. Training writes timestamped
run directories containing `config.json`, selected weights, predictions and
checkpoint-selection records. Evaluation adds `metrics.json`, turn-on plots
and `metrics_averaged.json` for compatible runs.
Keep the default truth-pT range 10--120 GeV and 44 bins for paper comparisons.
Do not replace per-seed evidence with fitted curve values.

## 5. FT using the recovered original weights

```text
python tools/single_object_artifacts.py restore-pretrained
python src/train.py --config configs/cnn_v12_ft_em2single_dual_fracbce/v12_ft_em2single_dual_fracbce_s42.json --data_dir "DATA_ROOT" --experiments_dir experiments/reproduced_best_ft
python src/evaluate.py --experiments_dir experiments/reproduced_best_ft --recalc
```

The first command verifies SHA-256 and restores the original `experiments/...`
paths expected by the configs, including each parent `config.json`. It does not
load a checkpoint into Python, train, or overwrite a different existing file.
No config edits are needed when using these recovered weights. Only load
PyTorch checkpoints from a trusted source; integrity hashes are not a sandbox.
Use the original dataset so the recomputed training-split normalization matches
the parent. Model weights alone are not a stand-alone detector inference service.

## 6. Optional: retrain the parent, then bind its new path

For E1+V4+F BCE+FT the BCE parent configs are:

| Seed | Parent config |
| --- | --- |
| 42 | `configs/cnn_v11_gpu07/v11_em2single_dual_frac_bce_s42.json` |
| 123 | `configs/cnn_v11_gpu08/v11_em2single_dual_frac_bce_s123.json` |
| 456 | `configs/cnn_v11_gpu09/v11_em2single_dual_frac_bce_s456.json` |

```text
python src/train.py --config configs/cnn_v11_gpu07/v11_em2single_dual_frac_bce_s42.json --data_dir "DATA_ROOT" --experiments_dir experiments/retrained_parents
python tools/bind_single_object_ft.py --config configs/cnn_v12_ft_em2single_dual_fracbce/v12_ft_em2single_dual_fracbce_s42.json --parent-run "ACTUAL_PARENT_RUN_DIRECTORY" --output results/generated/best_ft_s42.json
python src/train.py --config results/generated/best_ft_s42.json --data_dir "DATA_ROOT" --experiments_dir experiments/retrained_ft
```

Replace `ACTUAL_PARENT_RUN_DIRECTORY` with the timestamped directory produced
by the first command. The binding helper checks seed, features, architecture,
and checkpoint existence, and writes a **new** config without touching the
original. It cannot verify dataset identity; keep data, split and preprocessing
identical. E3+V2+F parents are
`configs/cnn_v11_gpu01/v11_em2stride_conv2x2_frac_s42.json`,
`cnn_v11_gpu02/..._s123.json`, and `cnn_v11_gpu03/..._s456.json` (all under
`configs/`); these parents use IF, not BCE.
For another FT config, its `initialization.weights_path` identifies the parent;
the recovered neighboring `config.json` is its exact saved training config.
If that parent is itself FT, follow its initialization dependency first.
Retraining is stochastic: expect statistical agreement, not identical weights.

## 7. Further single-object functionality

The [preserved original research README](SINGLE_OBJECT_RESEARCH_REFERENCE.md)
documents the configuration generator, feature registry, weighting profiles,
OR-budget search, constrained-training stages, CNN branches, and distribution
analysis. It is historical reference; this guide overrides old directory names,
notebook assumptions and ambiguous calibration wording. The notebooks are not
the supported final-paper reproduction entry point.

The five external normalizer files documented for the frozen **pair** candidate
are not required by this single-object training path. Both studies still need
the approved detector dataset for actual training/inference.
