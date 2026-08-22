import argparse
import json
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TauNet training configurations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory in which config JSON files are created.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate one lightweight CPU configuration instead of the full sweep.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        default=None,
        help=(
            "Comma-separated feature names. Repeat this argument to generate "
            "multiple feature combinations."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds for every requested feature set.",
    )
    parser.add_argument(
        "--checkpoint-method",
        action="append",
        choices=["validation_bce", "target_fpr"],
        default=None,
        help=(
            "Checkpoint selector. Repeat to retain both validation BCE and "
            "target-FPR checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-primary",
        choices=["validation_bce", "target_fpr"],
        default=None,
        help="Primary artifact method when both checkpoint selectors are used.",
    )
    parser.add_argument(
        "--checkpoint-target-fpr",
        type=float,
        default=0.005,
        help="Validation event FPR used by target_fpr selection (default: 0.005).",
    )
    parser.add_argument(
        "--classifier",
        choices=["nn_only", "tob_nn_or"],
        default=None,
        help="Final trigger classifier (default: legacy nn_only).",
    )
    parser.add_argument(
        "--classifier-target-fpr",
        type=float,
        default=0.005,
        help="Total event FPR for the classifier (default: 0.005).",
    )
    parser.add_argument(
        "--classifier-tob-fpr",
        type=float,
        default=0.004,
        help="TOB branch event-FPR budget for tob_nn_or (default: 0.004).",
    )
    parser.add_argument(
        "--classifier-tob-budget-mode",
        choices=["fixed", "validation_search"],
        default="fixed",
        help="Use a fixed TOB budget or select it on validation data.",
    )
    parser.add_argument(
        "--classifier-tob-budget-values",
        nargs="+",
        type=float,
        default=[0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025,
                 0.003, 0.0035, 0.004, 0.0045, 0.005],
        help="Candidate TOB event-FPR budgets for validation search.",
    )
    parser.add_argument(
        "--classifier-tob-budget-folds",
        type=int,
        default=2,
        help="Validation cross-fitting folds for TOB-budget search (default: 2).",
    )
    parser.add_argument(
        "--classifier-objective-min-pt",
        type=float,
        default=25.0,
        help="Lowest protected truth-pT value in GeV (default: 25).",
    )
    parser.add_argument(
        "--classifier-objective-max-pt",
        type=float,
        default=100.0,
        help="Upper truth-pT edge included in the mean objective (default: 100).",
    )
    parser.add_argument(
        "--classifier-objective-window-width",
        type=float,
        default=5.0,
        help="Truth-pT comparison-window width in GeV (default: 5).",
    )
    parser.add_argument(
        "--classifier-protected-max-pt",
        type=float,
        default=120.0,
        help="Upper truth-pT edge protected from regression (default: 120).",
    )
    parser.add_argument(
        "--classifier-noninferiority-mode",
        choices=[
            "per_window",
            "pooled_saturation",
            "multiscale_saturation",
        ],
        default="pooled_saturation",
        help=(
            "Use fine windows, one saturation pool, or overlapping statistical "
            "saturation guards (default: pooled_saturation)."
        ),
    )
    parser.add_argument(
        "--classifier-saturation-start-pt",
        type=float,
        default=60.0,
        help="Lower edge of the pooled saturation region in GeV (default: 60).",
    )
    parser.add_argument(
        "--classifier-noninferiority-tolerance",
        type=float,
        default=0.005,
        help="Allowed efficiency deficit per protected region (default: 0.005).",
    )
    parser.add_argument(
        "--classifier-saturation-window-width",
        type=float,
        default=30.0,
        help="Width of each statistical saturation window in GeV (default: 30).",
    )
    parser.add_argument(
        "--classifier-saturation-window-stride",
        type=float,
        default=10.0,
        help="Stride between statistical saturation windows in GeV (default: 10).",
    )
    parser.add_argument(
        "--classifier-no-full-saturation-pool",
        action="store_false",
        dest="classifier_include_full_saturation_pool",
        help="Disable the additional full saturation-region guard.",
    )
    parser.set_defaults(classifier_include_full_saturation_pool=True)
    parser.add_argument(
        "--classifier-uncertainty-mode",
        choices=["none", "paired_standard_error"],
        default="paired_standard_error",
        help="Uncertainty used by multiscale saturation guards.",
    )
    parser.add_argument(
        "--classifier-confidence-z",
        type=float,
        default=1.0,
        help="One-sided standard-error multiplier for saturation guards.",
    )
    parser.add_argument(
        "--classifier-allowed-physical-deficit",
        type=float,
        default=0.0,
        help="Allowed deficit after uncertainty protection (default: 0).",
    )
    parser.add_argument(
        "--classifier-objective-tie-tolerance",
        type=float,
        default=0.002,
        help="Objective difference treated as a tie (default: 0.002).",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "energy_weighted_bce", "constrained_trigger"],
        default=None,
        help="Training loss (default: legacy bce).",
    )
    parser.add_argument(
        "--constrained-initial-weights",
        type=str,
        default=None,
        help="Pretrained model weights used to initialize constrained fine-tuning.",
    )
    parser.add_argument(
        "--constrained-temperature",
        type=float,
        default=0.02,
        help="Smooth trigger temperature (default: 0.02 after surrogate audit).",
    )
    parser.add_argument(
        "--constrained-temperature-start",
        type=float,
        default=None,
        help="Optional starting temperature for continuation training.",
    )
    parser.add_argument(
        "--constrained-temperature-schedule",
        choices=["constant", "linear", "cosine"],
        default="constant",
        help="Temperature schedule used by constrained proxy cuts.",
    )
    parser.add_argument(
        "--constrained-primal-objective",
        choices=["soft_efficiency", "tail_ranking"],
        default="soft_efficiency",
        help="Differentiable primal objective (default: soft_efficiency).",
    )
    parser.add_argument(
        "--constrained-proxy-threshold-mode",
        choices=["fixed", "batch_rank"],
        default="fixed",
        help="Use a fixed legacy cut or a shift-invariant batch-rank proxy cut.",
    )
    parser.add_argument(
        "--constrained-objective-region",
        action="append",
        default=None,
        metavar="LOW,HIGH,WEIGHT",
        help="Objective-only truth-pT region. Repeat for multiple regions.",
    )
    parser.add_argument(
        "--constrained-constraint-region",
        action="append",
        default=None,
        metavar="LOW,HIGH,DEFICIT",
        help="Protected truth-pT region. Repeat for multiple regions.",
    )
    parser.add_argument(
        "--constrained-region",
        action="append",
        default=None,
        metavar="LOW,HIGH,WEIGHT,DEFICIT",
        help=(
            "Constrained truth-pT region in GeV with objective weight and "
            "allowed deficit. Repeat for multiple regions."
        ),
    )
    parser.add_argument(
        "--constrained-constraint-fraction",
        type=float,
        default=0.3,
        help="Fraction of training events reserved for dual updates (default: 0.3).",
    )
    parser.add_argument(
        "--constrained-dual-learning-rate",
        type=float,
        default=1.0,
        help=(
            "Legacy projected dual-ascent rate used for both constraint types "
            "unless a separate rate is provided (default: 1)."
        ),
    )
    parser.add_argument(
        "--constrained-fpr-dual-learning-rate",
        type=float,
        default=None,
        help="Optional projected dual-ascent rate for the event-FPR price.",
    )
    parser.add_argument(
        "--constrained-region-dual-learning-rate",
        type=float,
        default=None,
        help="Optional projected dual-ascent rate for energy-region prices.",
    )
    parser.add_argument(
        "--constrained-initial-fpr-multiplier-mode",
        choices=["fixed", "gradient_balance"],
        default="fixed",
        help=(
            "Initialize the FPR price from a fixed value or from training-only "
            "gradient scales (default: fixed)."
        ),
    )
    parser.add_argument(
        "--constrained-initial-fpr-multiplier",
        type=float,
        default=1.0,
        help="Fixed or fallback initial event-FPR price (default: 1).",
    )
    parser.add_argument(
        "--constrained-gradient-balance-batches",
        type=int,
        default=8,
        help="Training batches used for robust gradient balancing (default: 8).",
    )
    parser.add_argument(
        "--constrained-max-multiplier",
        type=float,
        default=10.0,
        help="Projection ceiling for constrained multipliers (default: 10).",
    )
    parser.add_argument(
        "--constrained-minimum-region-advantages",
        nargs="+",
        type=float,
        default=None,
        help="Minimum efficiency advantage over the baseline for each region.",
    )
    parser.add_argument(
        "--constrained-reference-model-deficits",
        nargs="+",
        type=float,
        default=None,
        help="Maximum efficiency loss from the pretrained model in each region.",
    )
    parser.add_argument(
        "--constrained-event-batch-size",
        type=int,
        default=512,
        help="Complete events per primal update (default: 512).",
    )
    parser.add_argument("--constrained-tail-fraction", type=float, default=0.05)
    parser.add_argument("--constrained-tail-temperature", type=float, default=0.2)
    parser.add_argument("--constrained-tail-min-events", type=int, default=16)
    parser.add_argument("--constrained-tail-memory-bank-size", type=int, default=0)
    parser.add_argument(
        "--constrained-fpr-violation-scale",
        type=float,
        default=1.0,
        help="Scale applied consistently to soft and hard FPR violations.",
    )
    parser.add_argument(
        "--loss-alpha",
        nargs="+",
        type=float,
        default=None,
        help="Alpha values generated for energy-weighted BCE.",
    )
    parser.add_argument(
        "--loss-power-p",
        nargs="+",
        type=float,
        default=None,
        help="Exponents p generated for continuous signal weights t^(-p).",
    )
    parser.add_argument(
        "--loss-power-pt-min",
        type=float,
        default=10.0,
        help="Lower truth-pT clamp for power-law weighting (default: 10 GeV).",
    )
    parser.add_argument(
        "--loss-power-pt-max",
        type=float,
        default=200.0,
        help="Upper truth-pT clamp for power-law weighting (default: 200 GeV).",
    )
    parser.add_argument(
        "--loss-include-inverse-frequency",
        action="store_true",
        help="Also generate a training-fitted inverse-frequency loss profile.",
    )
    parser.add_argument(
        "--loss-inverse-pt-min",
        type=float,
        default=25.0,
        help="Lowest truth-pT value used by inverse-frequency weighting.",
    )
    parser.add_argument(
        "--loss-inverse-pt-max",
        type=float,
        default=100.0,
        help="Upper truth-pT edge used by inverse-frequency weighting.",
    )
    parser.add_argument(
        "--loss-inverse-bin-width",
        type=float,
        default=5.0,
        help="Truth-pT bin width used by inverse-frequency weighting.",
    )
    parser.add_argument(
        "--loss-inverse-min-weight",
        type=float,
        default=0.2,
        help="Lower raw inverse-frequency weight limit.",
    )
    parser.add_argument(
        "--loss-inverse-max-weight",
        type=float,
        default=5.0,
        help="Upper raw inverse-frequency weight limit.",
    )
    return parser.parse_args()


def write_config(config, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return filepath


def generate_smoke_config(output_dir=DEFAULT_CONFIG_DIR):
    """Create one intentionally small end-to-end CPU test configuration."""
    config = {
        "run_id": "smoke_s42",
        "experiment_name": "Smoke_CPU_tob_pt",
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_layers": [16, 8],
        "features_to_use": ["tob_pt_only"],
        "epochs": 2,
        "seed": 42,
        "max_events_per_class": 5000,
    }
    filepath = write_config(config, Path(output_dir), "smoke_s42.json")
    print(f"Generated smoke-test configuration: {filepath}")


def generate_sweep_configs(
    output_dir=DEFAULT_CONFIG_DIR,
    feature_sets=None,
    seeds=None,
    checkpoint_selection=None,
    classifier=None,
    loss=None,
    loss_variants=None,
    initialization=None,
):
    base_config = {"epochs": 20}
    learning_rates = [0.001]
    batch_sizes = [256]
    architectures = [[32, 16]]
    if feature_sets is None:
        feature_sets = [
            ["tob_pt_only"],
            ["em2_3x3_maxdist", "tob_pt_only"],
            ["em2_3x3_dominance", "tob_pt_only"],
            ["em2_3x3_dominance", "em2_3x3_maxdist"],
            ["em2_3x3_dominance", "em2_3x3_maxdist", "tob_pt_only"],
        ]
    if seeds is None:
        seeds = [42, 123, 456]
    if loss_variants is None:
        loss_variants = [loss]

    output_dir = Path(output_dir)
    count = 0
    combinations = product(
        learning_rates,
        batch_sizes,
        architectures,
        feature_sets,
        loss_variants,
    )
    for config_number, (lr, bs, arch, features, loss_variant) in enumerate(
        combinations,
        start=1,
    ):
        constrained = (
            loss_variant is not None
            and loss_variant.get("name") == "constrained_trigger"
        )
        run_learning_rate = 0.0001 if constrained else lr
        features_str = "_".join(features)
        arch_str = "x".join(map(str, arch))
        experiment_name = (
            f"TauNet_lr{run_learning_rate}_bs{bs}_arch{arch_str}_{features_str}"
        )
        if loss_variant is not None and loss_variant.get("name") == "energy_weighted_bce":
            weighting = loss_variant["weighting"]
            if weighting["profile"] == "alpha":
                loss_tag = f"ewbce_a{weighting['alpha']:g}"
            elif weighting["profile"] == "power_law":
                loss_tag = f"ewbce_power_p{weighting['p']:g}"
            else:
                loss_tag = "ewbce_invfreq"
            experiment_name = f"{experiment_name}_{loss_tag}"
        elif loss_variant is not None and loss_variant.get("name") == "constrained_trigger":
            experiment_name = f"{experiment_name}_constrained"

        for seed in seeds:
            config = base_config.copy()
            if constrained:
                config["epochs"] = 10
            config.update(
                {
                    # Short IDs are used only for paths. Full metadata remains below.
                    "run_id": f"c{config_number:03d}_s{seed}",
                    "experiment_name": experiment_name,
                    "learning_rate": run_learning_rate,
                    "batch_size": bs,
                    "hidden_layers": arch,
                    "features_to_use": features,
                    "seed": seed,
                }
            )
            if checkpoint_selection is not None:
                config["checkpoint_selection"] = checkpoint_selection
            if classifier is not None:
                config["classifier"] = classifier
            if loss_variant is not None:
                config["loss"] = loss_variant
            if initialization is not None:
                config["initialization"] = initialization
            filename = f"c{config_number:03d}_s{seed}.json"
            write_config(config, output_dir, filename)
            count += 1

    print(f"Successfully generated {count} configuration files in '{output_dir}'")


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        generate_smoke_config(args.output_dir)
    else:
        requested_feature_sets = None
        if args.feature_set:
            requested_feature_sets = [
                [name.strip() for name in feature_set.split(",") if name.strip()]
                for feature_set in args.feature_set
            ]
            if any(not feature_set for feature_set in requested_feature_sets):
                raise ValueError("Each --feature-set must contain at least one feature name.")

        checkpoint_selection = None
        if args.checkpoint_method:
            primary = args.checkpoint_primary or args.checkpoint_method[0]
            if primary not in args.checkpoint_method:
                raise ValueError(
                    "--checkpoint-primary must also be supplied through "
                    "--checkpoint-method"
                )
            checkpoint_selection = {
                "methods": args.checkpoint_method,
                "primary_method": primary,
                "target_fpr": args.checkpoint_target_fpr,
            }

        classifier = None
        if args.classifier:
            classifier = {
                "name": args.classifier,
                "target_fpr": args.classifier_target_fpr,
                "trigger_objects": 2,
            }
            if args.classifier == "tob_nn_or":
                if args.classifier_tob_budget_mode == "fixed":
                    classifier["tob_fpr"] = args.classifier_tob_fpr
                else:
                    classifier["tob_budget"] = {
                        "mode": "validation_search",
                        "values": args.classifier_tob_budget_values,
                        "cross_validation_folds": args.classifier_tob_budget_folds,
                        "objective": {
                            "min_truth_pt_gev": args.classifier_objective_min_pt,
                            "objective_max_truth_pt_gev": args.classifier_objective_max_pt,
                            "window_width_gev": args.classifier_objective_window_width,
                            "protected_max_truth_pt_gev": args.classifier_protected_max_pt,
                            "noninferiority_mode": args.classifier_noninferiority_mode,
                            "saturation_start_truth_pt_gev": args.classifier_saturation_start_pt,
                            "noninferiority_tolerance": args.classifier_noninferiority_tolerance,
                            "saturation_window_width_gev": args.classifier_saturation_window_width,
                            "saturation_window_stride_gev": args.classifier_saturation_window_stride,
                            "include_full_saturation_pool": args.classifier_include_full_saturation_pool,
                            "uncertainty_mode": args.classifier_uncertainty_mode,
                            "confidence_z": args.classifier_confidence_z,
                            "allowed_physical_deficit": args.classifier_allowed_physical_deficit,
                            "objective_tie_tolerance": args.classifier_objective_tie_tolerance,
                        },
                    }
            elif args.classifier_tob_budget_mode != "fixed":
                raise ValueError(
                    "TOB-budget search requires --classifier tob_nn_or"
                )

        loss_variants = None
        initialization = None
        if args.loss == "bce":
            loss_variants = [{"name": "bce"}]
        elif args.loss == "energy_weighted_bce":
            requested_alphas = args.loss_alpha or []
            requested_powers = args.loss_power_p or []
            if any(alpha < 0 for alpha in requested_alphas):
                raise ValueError("--loss-alpha values must be non-negative")
            loss_variants = [
                {
                    "name": "energy_weighted_bce",
                    "weighting": {
                        "profile": "alpha",
                        "alpha": alpha,
                    },
                }
                for alpha in requested_alphas
            ]
            if not 0 < args.loss_power_pt_min < args.loss_power_pt_max:
                raise ValueError(
                    "Power-law pT limits must satisfy 0 < min < max"
                )
            loss_variants.extend(
                {
                    "name": "energy_weighted_bce",
                    "weighting": {
                        "profile": "power_law",
                        "p": power,
                        "pt_clip_min_gev": args.loss_power_pt_min,
                        "pt_clip_max_gev": args.loss_power_pt_max,
                    },
                }
                for power in requested_powers
            )
            if args.loss_include_inverse_frequency:
                loss_variants.append(
                    {
                        "name": "energy_weighted_bce",
                        "weighting": {
                            "profile": "inverse_frequency",
                            "pt_min_gev": args.loss_inverse_pt_min,
                            "pt_max_gev": args.loss_inverse_pt_max,
                            "bin_width_gev": args.loss_inverse_bin_width,
                            "min_weight": args.loss_inverse_min_weight,
                            "max_weight": args.loss_inverse_max_weight,
                        },
                    }
                )
            if not loss_variants:
                loss_variants = [
                    {
                        "name": "energy_weighted_bce",
                        "weighting": {"profile": "alpha", "alpha": 0.0},
                    }
                ]
        elif args.loss == "constrained_trigger":
            if not args.constrained_initial_weights:
                raise ValueError(
                    "--constrained-initial-weights is required for constrained training"
                )
            if args.classifier_tob_budget_mode != "fixed":
                raise ValueError(
                    "Initial constrained experiments require a fixed classifier budget"
                )
            raw_regions = args.constrained_region or [
                "25,40,0.3333333333333333,0.005",
                "40,60,0.3333333333333333,0.005",
                "60,120,0.3333333333333333,0.005",
            ]
            parsed_regions = []
            for raw_region in raw_regions:
                try:
                    low, high, weight, deficit = (
                        float(value.strip()) for value in raw_region.split(",")
                    )
                except ValueError as error:
                    raise ValueError(
                        "Each --constrained-region must be LOW,HIGH,WEIGHT,DEFICIT"
                    ) from error
                parsed_regions.append((low, high, weight, deficit))
            objective_regions = [
                (region[0], region[1], region[2]) for region in parsed_regions
            ]
            if args.constrained_objective_region:
                objective_regions = []
                for raw_region in args.constrained_objective_region:
                    try:
                        low, high, weight = (
                            float(value.strip()) for value in raw_region.split(",")
                        )
                    except ValueError as error:
                        raise ValueError(
                            "Each --constrained-objective-region must be LOW,HIGH,WEIGHT"
                        ) from error
                    objective_regions.append((low, high, weight))
            constraint_regions = [
                (region[0], region[1], region[3]) for region in parsed_regions
            ]
            if args.constrained_constraint_region:
                constraint_regions = []
                for raw_region in args.constrained_constraint_region:
                    try:
                        low, high, deficit = (
                            float(value.strip()) for value in raw_region.split(",")
                        )
                    except ValueError as error:
                        raise ValueError(
                            "Each --constrained-constraint-region must be LOW,HIGH,DEFICIT"
                        ) from error
                    constraint_regions.append((low, high, deficit))
            region_count = len(constraint_regions)
            for values, option in (
                (
                    args.constrained_minimum_region_advantages,
                    "--constrained-minimum-region-advantages",
                ),
                (
                    args.constrained_reference_model_deficits,
                    "--constrained-reference-model-deficits",
                ),
            ):
                if values is not None and len(values) != region_count:
                    raise ValueError(
                        f"{option} requires one value for each constrained region"
                    )
            loss_variants = [
                {
                    "name": "constrained_trigger",
                    "temperature_start": (
                        args.constrained_temperature
                        if args.constrained_temperature_start is None
                        else args.constrained_temperature_start
                    ),
                    "temperature_end": args.constrained_temperature,
                    "temperature_schedule": args.constrained_temperature_schedule,
                    "target_event_fpr": args.classifier_target_fpr,
                    "trigger_objects": 2,
                    "primal_objective": args.constrained_primal_objective,
                    "proxy_threshold_mode": (
                        args.constrained_proxy_threshold_mode
                    ),
                    "objective_regions_gev": [
                        [region[0], region[1]] for region in objective_regions
                    ],
                    "objective_region_weights": [
                        region[2] for region in objective_regions
                    ],
                    "constraint_regions_gev": [
                        [region[0], region[1]] for region in constraint_regions
                    ],
                    "allowed_deficits": [region[2] for region in constraint_regions],
                    "tail_fraction": args.constrained_tail_fraction,
                    "tail_temperature": args.constrained_tail_temperature,
                    "tail_min_events": args.constrained_tail_min_events,
                    "tail_memory_bank_size": (
                        args.constrained_tail_memory_bank_size
                    ),
                    "constraint_fraction": args.constrained_constraint_fraction,
                    "crossfit_folds": 2,
                    "fpr_violation_scale": args.constrained_fpr_violation_scale,
                    "fpr_dual_learning_rate": (
                        args.constrained_dual_learning_rate
                        if args.constrained_fpr_dual_learning_rate is None
                        else args.constrained_fpr_dual_learning_rate
                    ),
                    "region_dual_learning_rate": (
                        args.constrained_dual_learning_rate
                        if args.constrained_region_dual_learning_rate is None
                        else args.constrained_region_dual_learning_rate
                    ),
                    "initial_fpr_multiplier_mode": (
                        args.constrained_initial_fpr_multiplier_mode
                    ),
                    "initial_fpr_multiplier": args.constrained_initial_fpr_multiplier,
                    "gradient_balance_batches": (
                        args.constrained_gradient_balance_batches
                    ),
                    "max_multiplier": args.constrained_max_multiplier,
                    "event_batch_size": args.constrained_event_batch_size,
                }
            ]
            constrained_loss = loss_variants[0]
            if args.constrained_minimum_region_advantages is not None:
                constrained_loss["minimum_region_advantages"] = (
                    args.constrained_minimum_region_advantages
                )
            if args.constrained_reference_model_deficits is not None:
                constrained_loss["reference_model_allowed_deficits"] = (
                    args.constrained_reference_model_deficits
                )
            initialization = {
                "mode": "pretrained",
                "weights_path": args.constrained_initial_weights,
            }
            if classifier is None:
                classifier = {
                    "name": "nn_only",
                    "target_fpr": args.classifier_target_fpr,
                    "trigger_objects": 2,
                }

        generate_sweep_configs(
            output_dir=args.output_dir,
            feature_sets=requested_feature_sets,
            seeds=args.seeds,
            checkpoint_selection=checkpoint_selection,
            classifier=classifier,
            loss_variants=loss_variants,
            initialization=initialization,
        )
