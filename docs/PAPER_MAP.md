# Paper-to-code map

This page maps each table, figure and section of the manuscript to the run(s), module(s) and files that produce it.

- `outputs/experiments/<run>/` is written by `python -m src.experiments.<module> --project-root . --run-name <run>`.
- `results/paper_tables/` holds the tables used for the manuscript. `python -m src.reporting.paper_data` regenerates them into `outputs/paper/data/`, and `python -m src.reporting.paper_facts` regenerates `facts.json`.

Section numbers follow the version prepared for 计算机科学与探索 (2026-09-28): 1 Introduction, 2 Problem definition and framework, 3 Method, 4 Experimental design, 5 Results, 6 Discussion, 7 Conclusion. They may change before publication.

## Names used in the code

**Models**

| Paper | Onion code name | FMA / six-view code name |
|---|---|---|
| C1, Concat-H384/MD10 ("H384") | `FullConcatMD10-H384-BCE` | `ConcatMD10-H384` |
| C2, Concat-H192/MD30 ("H192") | `FullConcatMD-BCE` | `ConcatModDrop` |
| C3, RAMT (reliability-gated, H192, modality dropout 0.3) | `RAMT` | `RAMT` |
| MCE = 0.6 × C1 + 0.4 × C2 | `OptimizedEnsemble` (`MCE` inside the AA-MCE runs) | `MCE` |
| AA-MCE and its variants | `AA-MCE`, `AA-MCE-w`, `AA-MCE-2`, `MCE+AAT`, `X+AAT` (see docs/REPRODUCIBILITY.md) | same |
| AA-MCE with five candidates | `AA-MCE` in `availability_aware_ensemble_5c_v1` (`AA-MCE-5` in `ablation_summary.csv`) | – |

**Conditions**

| Paper | Onion / FMA code name | Six-view code name |
|---|---|---|
| Complete | `observed` | `observed` |
| −Audio | `no_audio` | `no_audio` |
| −Lyrics (FMA: −Text) | `no_lyrics` (`no_text`) | `no_lyrics` |
| −Visual (FMA: −Social) | `no_visual` (`no_social`) | `no_visual` |
| Random drop one | `random_one_missing` | `random_one_view_missing` |
| Random drop half | – | `random_half_views_missing` |
| Keep only one | `random_two_missing` | `keep_one_view` |

## Tables

| Table | Content | Runs | Modules | Files |
|---|---|---|---|---|
| Table 1 (§5.1, §5.2, §5.4) | Tagging performance and computational cost of all models on Music4All-Onion | `full_feature_ablation_v1` (single modalities, uniform mean, early concat, Concat+MD, reliability gate, RAMT); `optimized_multimodal_tagging_final_v1` (H192, H384, MCE); `advanced_fusion_baselines_v1` (AttnFusion, EvidFusion); `availability_aware_ensemble_v1` (AA-MCE); `paper_efficiency_v1` (parameters, latency) | `paper_data.py`, `paper_efficiency.py` | `results/paper_tables/onion_summary.csv`, `onion_tests.csv`; `outputs/experiments/paper_efficiency_v1/efficiency.csv` |
| Table 2 (§5.1, §5.3) | Ablation of the decision-layer components and controls with pattern-wise thresholds | `availability_aware_ensemble_v1`, `availability_aware_ensemble_5c_v1` | `availability_aware_ensemble.py`, `paper_data.py`, `paper_facts.py` | `ablation_summary.csv`, `aa_variant_tests.csv`, `aa5_tests.csv`, `facts.json` (key `five_vs_three`) |
| Table 3 (§5.5) | Transfer to FMA and to the six-view extension | `fma_cross_dataset_v1`, `availability_aware_ensemble_fma_v1`, `extended_modalities_v1`, `availability_aware_ensemble_sixview_v1` | `fma_cross_dataset.py`, `extended_modalities.py`, `availability_aware_campaigns.py`, `paper_data.py` | `fma_summary.csv`, `fma_aa_variants_summary.csv`, `fma_tests.csv`, `sixview_summary.csv`, `sixview_aa_variants_summary.csv`, `sixview_tests.csv`; `outputs/experiments/availability_aware_ensemble_{fma,sixview}_v1/paired_label_bootstrap_holm.csv` |

Contents of the test tables:

- `onion_tests.csv`, `fma_tests.csv` and `sixview_tests.csv` compare AA-MCE with every other model of the dataset. They use a label-level paired bootstrap (5,000 resamples) with Holm correction within condition × metric.
- `aa_variant_tests.csv` and `aa5_tests.csv` are copies of the AA-MCE runs' `paired_label_bootstrap_holm.csv`.

## Figures

All five manuscript figures are drawn by `python -m src.reporting.paper_figures --data <tables> --out <dir>`, which writes a vector PDF and a 600 dpi PNG. The labels are Chinese.

| Figure | Content | Inputs |
|---|---|---|
| Fig. 1 (§2.2, §3.2), `Fig1_framework_protocol` | AA-MCE framework and availability protocol | none (schematic; the manuscript uses a redrawn version of this layout) |
| Fig. 2 (§5.1, §5.2), `Fig2_main_results_robustness` | Complete-condition mAP of all models; mAP and Macro-F1 across conditions; missing-condition mean and worst case | `onion_summary.csv`, `ablation_summary.csv` |
| Fig. 3 (§5.3, §5.4), `Fig3_mechanism` | (a) RAMT gate weights; (b) validation mAP vs. modality dropout and capacity; (c) validation search of the two-component weight; (d) AA-MCE pattern weights | (a) `full_feature_ablation_v1/gate_weights.csv`; (b) `optimized_tagging_validation_v2/validation_sweep.csv`; (c) `optimized_tagging_validation_v2/validation_ensemble_search.csv`; (d) `availability_aware_ensemble_v1/pattern_weights.csv` (all under `outputs/experiments/`) |
| Fig. 4 (§5.5), `Fig4_transfer_fma_sixview` | FMA and six-view mAP / Macro-F1 across conditions | `fma_summary.csv`, `fma_aa_variants_summary.csv`, `sixview_summary.csv`, `sixview_aa_variants_summary.csv` |
| Fig. 5 (§5.6, §5.7), `Fig5_calibration_tag_semantics` | (a, b) ECE and Brier before and after pattern-wise Platt calibration; (c) AP per tag group; (d) per-tag AP vs. training frequency | (a, b) `outputs/experiments/posthoc_calibration_v1/calibration_metrics.csv`; (c) `tag_groups.csv`; (d) `tag_points.csv` |

Figures 1, 2 and 4 need only `results/paper_tables/`. Figures 3 and 5 also need the run outputs named above. The functions `figure_5` and `figure_6` in `paper_figures.py` are earlier layouts that `main()` does not call.

## Sections

| Section | Content | Code | Numbers |
|---|---|---|---|
| §2.1, §3.2 | Problem definition and availability-conditioned decision layer (Fig. 1) | `aa_core.py` (patterns, `simplex_grid`, `select_pattern_weights`, `fast_thresholds`, `evaluate_variants`), `availability_aware_ensemble.py`, `availability_aware_campaigns.py`, `posthoc_calibration.py` (Platt) | `facts.json`: `pattern_weights_onion`, `pattern_weights_fma`, `sixview_weight_gain_range`; per-pattern validation sizes: `eligible_validation_tracks` in each AA run's `run_manifest.json` |
| §3.1 | Candidate components | `optimized_multimodal_tagging.FullFeatureTagger` (C1, C2); `fusion_models.GenericTagger` via `full_feature_ablation.build_model("RAMT")` (C3) | – |
| §3.3 | Complexity of the offline selection and of inference | `aa_core.simplex_grid` (66 weight vectors for three components), `aa_core.THRESHOLD_GRID` (31 thresholds); `decision_layer_diagnostics.py` (selection timing) | `decision_layer_diagnostics_v1/selection_timing.json` |
| §4.1 | Datasets, artist-disjoint split, label sets, FMA social vector | `src/data/prepare.py` (`aa`, `features`, `tags`, `align`, `fma`), `robust_multimodal_tagging.build_dataset`, `fma_cross_dataset.build_task` / `build_split` | `numbers.json`: `onion_samples`, `onion_artists`, `onion_coverage`, `fma_samples`, `fma_artists`, `six_views`, `six_coverage_test`, `tag_frequencies_min_max`, `fma_echonest_fraction` |
| §4.2, §4.4 | Availability conditions, metrics and statistical inference | `intervention` in `robust_multimodal_tagging.py`, `fma_cross_dataset.py`, `extended_modalities.py`; `metric_values`, `summarise`; `common._bootstrap`; `full_feature_ablation.holm_by_family` | – |
| §4.3, §4.4 | Controls, bias control and reproducibility | `full_feature_ablation.py` (8 models), `advanced_fusion_baselines.py` (AttnFusion, EvidFusion), `fusion_models.py`, `configs/experiments.yaml` | `facts.json`: `epochs_E1F`, `elapsed_seconds` |
| §5.1 | Missing modalities stop ranking advantages from becoming decision advantages | Table 1, Table 2, Fig. 2c; `decision_layer_diagnostics.py` | `ablation_summary.csv`; same-run global-threshold values in `outputs/experiments/availability_aware_ensemble_v1/metrics_summary.csv`; `decision_layer_diagnostics_v1/threshold_shift_by_pattern.csv`, `validation_probability_by_pattern.csv`, `decision_loss_decomposition.csv` |
| §5.2 | The availability-conditioned decision layer corrects the mismatch | Table 1, Fig. 2; `robustness_checks.py` (track-level bootstrap) | `onion_summary.csv` (incl. `MicroF1`, `MacroROC_AUC`), `onion_tests.csv`, `aa_variant_tests.csv`; `robustness_checks_v1/track_level_bootstrap.csv`; `decision_layer_diagnostics_v1/observed_subset_map.csv`; `numbers.json`: `*_missing_mean_mAP`, `*_worst_mAP` |
| §5.3 | Two sources of gain: pattern-wise decisions and structural complementarity | Table 2, Fig. 3a, Fig. 3d; `full_feature_ablation.py` (mechanism tests); `robustness_checks.py` (δ sensitivity) | `facts.json`: `aa/*`, `gates`, `mechanism`; `robustness_checks_v1/selection_margin_sensitivity.csv` |
| §5.4 | More complex fusion structures bring no extra benefit | Table 1, Fig. 3b, Fig. 3c; five-candidate run; `paper_efficiency.py`; `decision_layer_diagnostics.py` (selection timing) | `facts.json`: `sweep`, `ensemble_selection`, `aa5/AA-MCE`, `five_vs_three`, `efficiency`; `paper_efficiency_v1/efficiency.csv`; `decision_layer_diagnostics_v1/selection_timing.json` |
| §5.5 | The correction transfers but fails in low-information patterns | Table 3, Fig. 4 | `three_vs_six_tests.csv`, `fma_tests.csv`, `sixview_tests.csv`; `outputs/experiments/paper_analysis_v1/aamce_transfer_vs_frozen_paired_holm.csv` |
| §5.6 | Probability reliability also needs availability conditioning (Onion only) | Fig. 5a, Fig. 5b; `posthoc_calibration.py` | `outputs/experiments/posthoc_calibration_v1/calibration_metrics.csv`, `calibration_paired_holm.csv`; `facts.json`: `calibration`, `calibration_tests` |
| §5.7 | Semantic boundary of content-minable tag knowledge | Fig. 5c, Fig. 5d; `paper_analysis.py` (tag groups) | `tag_groups.csv`, `tag_points.csv`, `tag_group_tests.csv`; `facts.json`: `tag_groups/*`, `tags/*` |

## Runs

| Run | Command (from the repository root) | Main outputs |
|---|---|---|
| `robust_multimodal_tagging_v3` | `python -m src.experiments.robust_multimodal_tagging --project-root . --run-name robust_multimodal_tagging_v3` | compact-feature reference family; `labels.json`; mean test predictions |
| `optimized_tagging_validation_v2` | `python -m src.experiments.optimized_multimodal_tagging --project-root . --run-name optimized_tagging_validation_v2 --stage sweep`, then `python -m src.experiments.select_tagging_ensemble --project-root . --run-name optimized_tagging_validation_v2` | `validation_sweep.csv`, `selection.json`, `validation_ensemble_search.csv`, `ensemble_selection.json` |
| `optimized_multimodal_tagging_final_v1` | `... optimized_multimodal_tagging ... --run-name optimized_multimodal_tagging_final_v1 --stage final` | C1/C2 checkpoints, MCE metrics, thresholds, mean predictions |
| `advanced_fusion_baselines_v1` | `... advanced_fusion_baselines --run-name advanced_fusion_baselines_v1` | AttnFusion, EvidFusion; `efficiency.csv` |
| `full_feature_ablation_v1` | `... full_feature_ablation --run-name full_feature_ablation_v1` | eight feature-matched models incl. RAMT; `gate_weights.csv`; `efficiency.csv` |
| `fma_cross_dataset_v1` | `... fma_cross_dataset --run-name fma_cross_dataset_v1` | FMA models and MCE |
| `extended_modalities_v1` | `... extended_modalities --run-name extended_modalities_v1` | six-view models and MCE |
| `availability_aware_ensemble_v1` | `... availability_aware_ensemble --run-name availability_aware_ensemble_v1` | AA-MCE (Onion): `pattern_weights.csv`, `pattern_weight_grid.csv`, `pattern_thresholds.csv`, variants, tests, checkpoints |
| `availability_aware_ensemble_5c_v1` | `... availability_aware_ensemble --section availability_aware_ensemble_5c --run-name availability_aware_ensemble_5c_v1` | five-candidate upper bound |
| `availability_aware_ensemble_fma_v1` | `... availability_aware_campaigns --dataset fma --run-name availability_aware_ensemble_fma_v1` | AA-MCE on FMA |
| `availability_aware_ensemble_sixview_v1` | `... availability_aware_campaigns --dataset sixview --run-name availability_aware_ensemble_sixview_v1` | AA-MCE on six views (63 patterns) |
| `posthoc_calibration_v1` | `... posthoc_calibration --run-name posthoc_calibration_v1` | `calibration_metrics.csv`, `calibration_paired_holm.csv` |
| `robustness_checks_v1` | `... robustness_checks --run-name robustness_checks_v1` | `track_level_bootstrap.csv`, `selection_margin_sensitivity.csv` |
| `paper_analysis_v1` | `... paper_analysis --run-name paper_analysis_v1` | frozen-asset check, calibration audit, robustness aggregates, FMA natural-missingness strata, tag groups, AA-MCE transfer tests |
| `paper_efficiency_v1` | `python -m src.experiments.paper_efficiency --project-root .` | `efficiency.csv` (GPU and single-thread CPU), pattern lookup time |
| `decision_layer_diagnostics_v1` | `... decision_layer_diagnostics --run-name decision_layer_diagnostics_v1` (inference from saved checkpoints; `--output-root` writes elsewhere) | `threshold_shift_by_pattern.csv`, `validation_probability_by_pattern.csv`, `decision_loss_decomposition.csv`, `observed_subset_map.csv`, `selection_timing.json` |

## Extended figure and table suites

`final_figures.py` and `final_tables.py` produce an English figure and table suite for supplementary material from the same runs:

- `final_figures.py` writes Fig_1 … Fig_14, Fig_S1 and `figure_captions.md` to `outputs/figures_final`.
- `final_tables.py` writes Table_1 … Table_16 and Table_S1, as CSV and Markdown, to `outputs/tables_final`.

Their numbering does not follow the manuscript. They use the same results and add the compact-feature comparison (Fig. S1, Table S1) and the FMA natural-missingness strata.
