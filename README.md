# Tau Particle Neural Network Classifier

## Overview
This project provides an automated machine learning framework designed to build, train and test different Neural Networks for Tau particle identification.
These neural nets are trained on simulated particle collision data, containing both background noise and signal data.
In essence, it is built to automatically train many different neural nets, changing the input parameters and the hidden layer structure, in order to find the potimal neural net for tau identification.
Our workflow involved deciding on a "batch" of neural nets configurations, and then using the entire workframe to build, train, test and analyze neural nets with all those configurations.
---

## Dependencies
torch, numpy, pandas, scipy, tqdm and pyarrow.

default modules (preinstalled): os, json, itertools, argparse, datetime, random, glob, collections, copy.


For Report_Plots jupyter notebook analysis: matplotlib, seaborn.

The code attempts to utilize CUDA (Nvidia GPU interface) to improve performance. 
## Directory Structure

```text
.
├── Background/                  # Simulated background data (.csv and .npz files)
├── Signal/                      # Simulated tau signal data (.csv and .npz files)
├── configs/                     # Automatically generated JSON files that determine a neural net training parameters (input parameters, structure, training hyperparameters etc)
│   └── distributions/           # JSON definitions for distribution analyses
├── experiments/                 # Contains "experiment" folders. Every trained neural net has an "experiment" folder with the model weights, predictions, and metrics. Currently contains all trained neural nets organized by "batches".
├── analysis_outputs/            # Generated research-analysis outputs (ignored by Git)
│   └── distributions/           # Feature histograms and split-audit tables
├── Reserach results data/       # Analyzed neural net results (graphs, testing evaluation)
├── src/                         # Main code files. Use with --help flag for detailed usage explanation.
│   ├── distributions/           # Modular object/event distribution analysis and split auditing
│   ├── evaluate.py              # Generates evaluation results for trained neural nets (Fake Rate calculations, binned efficiencies, and Fermi-Dirac fitting)
│   ├── features.py              # Input parameters feature engineering
│   ├── model.py                 # PyTorch implementation of the Dynamic MLP
│   ├── tracker.py               # Experiment tracking and artifact archiving
│   ├── training_data.py         # Reusable data loading, alignment, event splitting, and in-memory caching
│   ├── operating_point.py       # Shared event-level FPR threshold calculations
│   ├── classifiers.py           # Configurable NN-only and TOB-NN OR trigger decisions
│   ├── classifier_selection.py  # Cross-fitted validation search for the TOB branch budget
│   ├── losses.py                # Configurable training-loss construction
│   ├── checkpoint_selection.py  # Configurable validation checkpoint selection
│   ├── constrained_objective.py # Smooth and exact constrained trigger metrics
│   ├── constrained_training.py  # Event-level primal/dual fine-tuning loop
│   ├── constrained_diagnostics.py # Pre-training hard/soft surrogate audit
│   ├── event_data.py            # Complete-event batching and inner training split
│   └── train.py                 # Main training loop and configuration-sweep orchestration
├── tests/                       # Automated correctness checks for reusable project logic
│   ├── test_distributions.py    # Distribution, event grouping, and split regression tests
│   ├── test_evaluate_thresholds.py # Event-level threshold calibration regression tests
│   ├── test_features.py         # Object-specific geometry and feature regression tests
│   ├── test_checkpoint_selection.py # Validation checkpoint-selection regression tests
│   ├── test_classifiers.py      # Classifier calibration, FPR, and composition tests
│   ├── test_classifier_selection.py # TOB-budget search and validation-fold tests
│   ├── test_losses.py           # Energy-weighted loss and normalization tests
│   ├── test_constrained_objective.py # Trigger-surrogate regression tests
│   ├── test_constrained_training.py # Primal/dual update regression tests
│   ├── test_event_data.py       # Complete-event batching and leakage tests
│   └── test_training_data_cache.py # Cache equivalence, determinism, and leakage regression tests
├── slurm/
│   └── run_config_batch.slurm   # Reusable GPU runner for a generated configuration batch
├── generate_configs.py          # Generates JSON config sweeps from feature sets, seeds, and hyperparameters
├── requirements.txt             # Python dependencies required by the project
├── Report_Plots.ipynb           # Aggregation and plotting notebook for experiment results
├── Report_Plots - Batch 3...    # Focused evaluation notebook for later experiment batches
├── exp_results_table.csv        # Temporary file generated by Report_Plots (compares many neural nets testing results).
```

## Usage Workflow

*Run all the code from "Refactored Codebase" relative path.*

### Step 0: Designing a testing batch

Decide on what testing you want to do - specifically, which features (the neural net input vector parameters) and what hidden layer architecture.

Feature distributions can optionally be generated before designing a batch:

```bash
python -m src.distributions.run --config configs/distributions/core_v1.json
```

This produces object-level, event-level and pT-conditioned histograms under `analysis_outputs/distributions/`. The same framework can audit the 70/10/20 event split over multiple seeds.

### Step 1: Generate Configuration Sweep
The generation script creates lists of values for: epoches amount, learning rates, batch sizes, architectures, and feature_ests and seeds.
It then creates all possible combinations of those lists to automatically generates config JSON files for every neural net you want testing.
See examples in previous testing batches under configs/ folder. 

Configuration filenames use short IDs such as `c001_s42.json`. The full feature list,
architecture, hyperparameters, and descriptive experiment name remain inside the JSON,
so no metadata is lost and generated paths remain portable.

```bash
python generate_configs.py
```
*This populates the `configs/` folder with uniquely named config files.*

For one lightweight CPU smoke-test configuration:

```bash
python generate_configs.py --smoke-test
```

Custom feature sets and seeds can be supplied without editing the script. Commas combine features within one network, while repeating `--feature-set` creates separate experiment families:

```bash
python generate_configs.py --output-dir configs/event_features_v1 \
  --feature-set "event_sum_tob_pt" \
  --feature-set "event_second_highest_tob_pt" \
  --feature-set "em2_normalized_width,event_sum_tob_pt" \
  --seeds 42 123 456
```

This example creates nine configurations: three feature sets evaluated with three random seeds. The available generator options are `--feature-set`, `--seeds`, `--output-dir`, and `--smoke-test`.

Checkpoint selection can use validation BCE, signal efficiency at a target
event FPR, or both methods in the same training run:

```bash
python generate_configs.py --output-dir configs/checkpoint_study \
  --feature-set "core_physics,em2_best_3x3_fraction" \
  --seeds 42 123 456 \
  --checkpoint-method validation_bce \
  --checkpoint-method target_fpr \
  --checkpoint-primary target_fpr \
  --checkpoint-target-fpr 0.005
```

The final trigger classifier and training loss are configured independently.
Legacy configs default to `nn_only` with BCE. The hybrid classifier accepts an
object when either its TOB pT or its network score passes the calibrated cut:

```bash
python generate_configs.py --output-dir configs/or_study \
  --feature-set "em0_sum,em1_sum,em2_3x3_sum,em3_sum,had_sum" \
  --seeds 42 123 456 \
  --classifier tob_nn_or \
  --classifier-target-fpr 0.005 \
  --classifier-tob-fpr 0.004 \
  --loss bce \
  --checkpoint-method target_fpr
```

The TOB budget can also be selected independently for every feature set. The
search uses two event-level validation folds and never uses test data:

```bash
python generate_configs.py --output-dir configs/or_budget_search \
  --feature-set "em0_sum,em1_sum,em2_3x3_sum,em3_sum,had_sum" \
  --seeds 42 123 456 \
  --classifier tob_nn_or \
  --classifier-tob-budget-mode validation_search \
  --checkpoint-method target_fpr
```

Candidate budgets and objective limits can be changed through the
`--classifier-tob-budget-*` and `--classifier-objective-*` options.
By default, noninferiority is checked in 5-GeV windows from 25 to 60 GeV and
in one pooled 60--120 GeV saturation region. Fine high-pT windows remain in
the saved diagnostics but cannot reject a candidate through a single sparse
bin. The protection mode, saturation edge, upper edge, and tolerance are all
configurable; `per_window` reproduces the earlier fine-window rule.

`multiscale_saturation` keeps the fine protection below 60 GeV and checks the
saturation range with overlapping 30-GeV windows plus the full 60--120 GeV
pool. Its lower guard is `delta - z * standard_error`, where the paired error
is clustered by event. Window width, stride, `z`, and the allowed physical
deficit are configurable:

```bash
python generate_configs.py --output-dir configs/or_multiscale \
  --feature-set "core_tensors" \
  --seeds 42 123 456 \
  --classifier tob_nn_or \
  --classifier-tob-budget-mode validation_search \
  --classifier-noninferiority-mode multiscale_saturation \
  --classifier-saturation-window-width 30 \
  --classifier-saturation-window-stride 10 \
  --classifier-confidence-z 1.0 \
  --classifier-allowed-physical-deficit 0.0 \
  --checkpoint-method target_fpr
```

Energy-weighted BCE can generate several alpha values and a training-fitted
inverse-frequency profile in the same sweep:

```bash
python generate_configs.py --output-dir configs/weighted_bce_screen \
  --feature-set "core_tensors" \
  --seeds 42 \
  --loss energy_weighted_bce \
  --loss-alpha 0 1 2 4 \
  --loss-include-inverse-frequency
```

Every alpha is a separate full training run. The inverse-frequency profile uses
5-GeV signal bins from 25 to 100 GeV by default; its range, bin width, and weight
limits can be changed through the `--loss-inverse-*` options.

A continuous power-law profile applies `clip(truth_pt, 10, 200)^(-p)` to signal
objects and normalizes the mean training-signal weight to one. Background weights
remain one. Multiple exponents can be generated without editing code:

```bash
python generate_configs.py --output-dir configs/power_law_screen \
  --feature-set "core_tensors" \
  --seeds 42 \
  --loss energy_weighted_bce \
  --loss-power-p -1 -0.6 -0.3 0 0.3 0.6 1
```

The clamp can be changed with `--loss-power-pt-min` and
`--loss-power-pt-max`. Here `p=0` is exactly ordinary BCE, negative values
emphasize higher-pT signal, and positive values emphasize lower-pT signal.

Direct constrained fine-tuning is selected with `loss.name` set to
`constrained_trigger`. It starts from an existing checkpoint, keeps complete
events together, and optimizes smooth signal efficiency under event-FPR and
coarse energy noninferiority constraints. The training split is divided by
event into independent primal and constraint subsets; validation still selects
the checkpoint and test remains untouched. Example NN-only and fixed-budget OR
configs are under `configs/constrained_stage_d_nn_s42/` and
`configs/constrained_stage_e_or_s42/`. The gradient-balanced NN-only follow-up
is under `configs/constrained_stage_d2_nn_gradbalance_s42/`. The controlled
multiplier and reference-guard comparison is under
`configs/constrained_stage_d3_nn_s42/`. The corrected saturation guard and
faster energy-price response are under `configs/constrained_stage_d4_nn_s42/`.

The event-FPR multiplier can use the legacy fixed initialization or
`initial_fpr_multiplier_mode: "gradient_balance"`. Gradient balancing measures
the objective and FPR gradient norms on training batches only, uses their median
ratio, and clips it to the configured multiplier limit. Each epoch records soft
and hard metrics, gradient scales, their cosine similarity, and
signal/background score quantiles. The best checkpoint remains the primary
output, while `last_epoch_weights.pt` is saved only for debugging failed or
unstable training.

The event-FPR and energy-region prices can use separate ascent rates through
`fpr_dual_learning_rate` and `region_dual_learning_rate`. The legacy
`dual_learning_rate` still sets both values when the separate fields are
omitted. Training also reports an initial warning when an efficiency guard has
less slack than one observed signal object in its region.

Energy guards can also protect the pretrained model. For region `k`, the
required efficiency is the larger of `baseline + minimum_region_advantages[k]`
and `pretrained - reference_model_allowed_deficits[k]`. Omitting these fields
keeps the legacy baseline-deficit behavior. The reference network is frozen and
uses the same training-only calibration, so it cannot leak validation or test
information into gradient updates.

Before training, saved predictions can be used to verify that smooth and exact
trigger metrics behave consistently across several temperatures:

```bash
python src/constrained_diagnostics.py \
  experiments/nn_only_core_fraction_s42 \
  --classifier nn_only \
  --temperatures 0.02 0.05 0.1
```

This is an engineering audit only; test predictions must not select scientific
hyperparameters. New constrained configs can also be generated with
`--loss constrained_trigger`, `--constrained-initial-weights`, repeated
`--constrained-region LOW,HIGH,WEIGHT,DEFICIT`, and the other
`--constrained-*` options. Reference guards can be generated with
`--constrained-minimum-region-advantages` and
`--constrained-reference-model-deficits`; the multiplier ceiling is controlled
by `--constrained-max-multiplier`. Separate dual rates can be generated with
`--constrained-fpr-dual-learning-rate` and
`--constrained-region-dual-learning-rate`.

### Step 2: Train the Network
Main training script. 
Standard usage is giving it a config directory, and it will generate an "experiment" folder for each JSON config file within that directory.

```bash
python src/train.py --configs_dir configs/
```

During a configuration sweep, aligned data, event split indices, and deterministic raw features are reused in memory. Each experiment still calculates its own training-set normalization and creates a new model, optimizer, and prediction output.

The raw-feature cache is bounded to 512 MB by default. Its size can be changed with:

```bash
python src/train.py --configs_dir configs/ --feature_cache_mb 256
```

Caching can be disabled for reproducibility checks:

```bash
python src/train.py --configs_dir configs/ --disable_data_cache
```

A generated batch can be trained, evaluated, and archived on Slurm with the
reusable runner. Create `logs/` before submitting because Slurm opens its log
files before the job starts:

```bash
mkdir -p logs
sbatch --export=ALL,CONFIG_BATCH=weighted_bce_pooled_s3 \
  slurm/run_config_batch.slurm
```

To train only the CPU smoke test:

```bash
python src/train.py --config configs/smoke_s42.json
```
*Artifacts, including `config.json`, `model_weights.pt`, and predictions, are safely archived into timestamped subfolders within `experiments/`. Predictions use Parquet when available and automatically fall back to CSV when a Parquet engine is blocked or unavailable.*

When both checkpoint methods are enabled, the primary method keeps the standard
artifact names. The secondary method receives a method suffix.
`checkpoint_selection.json` records the best epochs, validation threshold,
achieved FPR, signal efficiencies, classifier calibration, and selected loss.

### Step 3: Run Physics Evaluation
Extract event-level thresholds, calculate efficiencies, and perform Fermi-Dirac curve fitting.
Generates a `metrics.json` file containing both the requested and actually achieved
background fake rates within every experiment folder.
The main turn-on curve recalibrates both the configured classifier and the TOB
baseline on the evaluated background sample. This gives both curves the same
requested event FPR. For trained checkpoints, `metrics.json` also keeps the
validation-calibrated operating point as a separate generalization diagnostic.
By default, it scans the project-local `experiments/` directory and skips runs that
already contain `metrics.json`. Each evaluated run also receives `turn_on_curve.png`.
```bash
python src/evaluate.py
```

The hybrid classifier can also be explored on existing predictions without
retraining. These post-hoc results use a separate suffix and do not overwrite
the original metrics. Their thresholds use the same test recalibration policy as
the standard turn-on comparison:

```bash
python src/evaluate.py --experiments_dir experiments/batch_name \
  --classifier tob_nn_or --classifier-tob-fpr 0.004 --recalc
```

Hybrid evaluation saves two turn-on plots: the configured classifier against
the TOB baseline, and a separate branch diagnostic showing NN and TOB branches.

### Step 4: Analyze Results
Launch Jupyter Notebook to aggregate and visualize the findings.
Both Report Plots notebooks start by loading all the results of a specific experiments batch folder, and saving a "exp_results_table.csv" temporary file showcasing the results and fit params.
The rest of the notebooks cells are edits frequently depending, so we can't document them - but generally, they are used to select specific neural nets and graphing out signal identification efficiency as a function of truth_pt.

*We manually add a folder named run_0_baseline_tob_pt to every experiment's batch folder. It simply has the graphing results for the baseline tob_pt graph (datapoint and fit).*
*This enables the Report Plots to easily load that baseline tob pt graph with all the other experiments for comparison.*

```bash
jupyter notebook Report_Plots.ipynb
```


## Core Components

### 1. Data Pipeline & Neural Net Training (`src/training_data.py`, `src/train.py`)
The data pipeline ingests both CSV metadata and NPZ tensor arrays from the `Signal/` and `Background/` directories.  
* **Splitting:** The dataset is split into training (70%), validation (10%), and testing (20%) subsets. Crucially, this split is performed by isolating unique event IDs to prevent data leakage across objects within the same collision event.
* **In-memory reuse:** During multi-configuration runs, deterministic data alignment, event splits, and raw features are reused to avoid repeated preparation work. The cache is discarded when the Python process ends.
* **Experiment independence:** Normalization is calculated only from each training split. Models, optimizers, predictions, and evaluation results are never shared between experiments.
* **Reproducibility:** Random seeds are reset before every model is constructed, so cache hits and configuration order do not change training results.

### 2. Feature Engineering (`src/features.py`)
The framework leverages a feature registry to dynamically extract variables for the input vector. Key physics variables include:
The config files uses the registry naming scheme to decide which specific features it wants to give for the input vectors, and the trian.py code uses this registry for name resolution.
Each feature is defined as a function that receives a dataframe containing all the data, and can parse out exactly what it wants. 
* **Important Note:** The returned value from each feature function is a list of results, as it calculates the feature for the entire batch at once. 
* **Normalized shower features:** `em2_normalized_width`, `em2_3x3_normalized_dominance`, `em2_3x3_sum_over_tob_pt`, and `em2_best_3x3_fraction` use ordinary floating-point division. Exact zero denominators are mapped to zero to avoid invalid model inputs.
* **Event context:** `event_sum_tob_pt`, `event_second_highest_tob_pt`, and `event_top2_tob_dr` are calculated from observable TOB data and broadcast to every object in the event. `event_context_core` returns all three values together.
* **Object-specific event geometry:** `object_reference_tob_dr2` returns squared angular distance to the highest-`tob_pt` object. The leading object instead uses the second-highest object as its reference. `object_partner_context` returns log object and partner `tob_pt`, azimuthal acoplanarity, and absolute eta separation using the same partner rule.
* **Leakage protection:** Event features do not use truth labels or tau multiplicity. Background event IDs are separated from signal IDs, and complete events remain within one train, validation, or test subset.

### 3. Model Architecture (`src/model.py`)
Classification is handled by `DynamicMLP`, a modular Multi-Layer Perceptron built with `torch.nn`.  
* The architecture concatenates the requested features into an initial input dimension and constructs the hidden layers dynamically based on the configuration array (e.g., `[32, 16]`).  
* It utilizes ReLU activations for hidden layers and a final Sigmoid activation for binary classification.  

### 4. Classifiers and Losses (`src/classifiers.py`, `src/losses.py`)
* **Independent configuration:** The final classifier and training loss use separate config sections, so every classifier can be combined with future losses.
* **Legacy classifier:** `nn_only` applies the calibrated network-score threshold and remains the default for old configs.
* **Hybrid classifier:** `tob_nn_or` accepts each object when either the TOB-pT branch or NN branch passes. Both thresholds are calibrated at event level, including mixed events where each branch contributes one accepted object.
* **TOB-budget search:** `validation_search` jointly selects the checkpoint and TOB budget. Complete events stay together in deterministic cross-fitting folds. The objective maximizes the mean OR-minus-baseline efficiency in 5-GeV windows from 25 to 100 GeV. Noninferiority protects fine windows below 60 GeV and one pooled 60--120 GeV saturation region by default, avoiding vetoes from statistically sparse high-pT windows.
* **Validation safety:** During training, classifier thresholds and target-FPR checkpoints use validation data only. The calibrated decision is then fixed for test evaluation.
* **Energy-weighted BCE:** `energy_weighted_bce` supports fixed alpha, training-fitted inverse-frequency, and continuous power-law profiles. Only signal examples are redistributed across truth-pT regions; background weights remain one.
* **Weight normalization:** Signal weights are divided by their mean on the training split, preserving the total signal contribution. Inverse-frequency weights are bounded and power-law pT values are clamped before normalization to limit gradient variance. The fitted profile is reused unchanged on validation data and saved in `checkpoint_selection.json`.
* **Direct constrained objective:** `constrained_trigger` replaces hand-written energy weights with a smooth event-level objective. Network weights descend on the surrogate while non-negative multipliers ascend from exact measurements on a separate training-event subset. NN-only and fixed-budget OR classifiers use the same implementation through config.

### 5. Evaluation & Metrics (`src/evaluate.py`)
The evaluation script operates on the `predictions.parquet` files generated during testing.  
* Converts kinematic constraints (like `tob_pt` and `truth_pt`) from MeV to GeV.  
* Calibrates each score threshold at the event level. For the two-object trigger, an event passes when at least two objects satisfy `score >= threshold`; equivalently, the threshold is compared with the event's second-highest object score.
* Selects the lowest deterministic threshold whose measured background Fake Rate does not exceed the requested target (e.g., 0.005, 0.010, 0.020). Equal scores are kept together, so a discrete model may achieve a lower Fake Rate when the exact target is unattainable.
* Applies threshold calibration and efficiency cuts with the same float64 `>=` comparison, including for tied discrete scores, and saves the measured rate under `achieved_fake_rates` in `metrics.json`.
* Measures efficiency only on truth-matched tau objects (`Type == "Signal"` and `signal == 1`), excluding noise objects contained in signal events.
* Bins the truth-matched tau data across 44 segments to calculate binomial efficiencies and maps them to a Fermi-Dirac function via `scipy.optimize.curve_fit` to extract the midpoint, slope, and plateau.
* Adds a black `tob_pt` baseline to every turn-on plot, calibrated independently on the same test events and target Fake Rate.
* Multi-seed experiments are automatically aggregated, including their independently calibrated `tob_pt` baselines, to compute stable statistical metrics.
* Optional checkpoint variants are evaluated separately. Target-FPR checkpoints also report test performance at the fixed threshold calibrated only on validation data.
