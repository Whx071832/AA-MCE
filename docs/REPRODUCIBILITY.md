# Reproducibility

This page summarises the reproducibility settings of the code: seeds, splits, masks, how every tuned quantity is selected, what the runs depend on, and how long they take. All values come from `configs/experiments.yaml` and the modules named below.

## Environment

The paper was produced on Windows 11 with the following setup:

- Python 3.13.5 and PyTorch 2.12.0+cu126;
- one NVIDIA RTX 3090 (24 GB), an AMD Ryzen 5 7500F and 32 GB RAM.

`requirements.txt` pins the package versions. Each run records its environment in `run_manifest.json`: Python, PyTorch and CUDA versions, device and GPU name.

## Seeds and determinism

| What | Seed |
|---|---|
| Model training (every five-seed campaign) | 20260724, 20260725, 20260726, 20260727, 20260728 (`seeds:`) |
| Validation sweep of the MCE candidates | 20260724 (`sweep_seed`) |
| Availability masks | 20260910 + k, where k is the position of the condition in the dataset's condition list |
| Label-level bootstrap | fixed base seed per analysis + running index of the test (e.g. 20261101 for Onion AA-MCE, `bootstrap_seed` 20261301 for FMA and 20261401 for six views, 20261901 / 20262001 / 20262101 in `paper_data.py`) |
| Track-level bootstrap | 20261501 + comparison index (Onion), 20261601 + comparison index (FMA) |
| Efficiency audits | `torch.manual_seed(20260724)` |
| Compact-feature reference family | CountSketch projections seeded by `stable_int("RAMT/countsketch/<view>/<dimension>")` |

`set_seed(seed, deterministic=True)` (`src/experiments/common.py`) does four things:

- seeds NumPy and PyTorch (CPU and all GPUs);
- calls `torch.use_deterministic_algorithms(True, warn_only=True)`;
- disables cuDNN benchmarking;
- the mini-batch order then comes from `np.random.default_rng(seed)`.

When CUDA is available, importing `src/experiments/fusion_models.py` disables the fused flash and memory-efficient attention kernels, so the math backend is used.

On the authors' workstation, re-running with these seeds reproduced the frozen predictions exactly. Every `reproduction_check.json` of the four AA-MCE runs reports a maximum absolute difference of 0.0 between the recomputed and the frozen mean test predictions, for MCE, RAMT, AttnFusion and EvidFusion as applicable. Which models are recomputed and which are reloaded differs by run:

- `availability_aware_ensemble_v1` reloads the MCE components from the frozen checkpoints and retrains AttnFusion and EvidFusion.
- The five-candidate run reloads the checkpoints of `availability_aware_ensemble_v1`.
- The FMA and six-view runs retrain every compared model.

On other GPUs, drivers or library versions small numerical differences are expected: `warn_only=True` allows non-deterministic kernels, with a warning.

## Artist-disjoint splits (BLAKE2b hashing)

- **Music4All-Onion.** The `align` stage of `src/data/prepare.py` assigns every Music4All-A+A artist id a bucket, `stable_bucket(artist_id, 10)`. This is the 8-byte BLAKE2b digest of the UTF-8 id, read as a big-endian integer, modulo 10.
  - Buckets 0-7 go to train, bucket 8 to validation and bucket 9 to test.
  - Tracks inherit the split of their artist through the A+A track-artist edges. Onion tracks without an A+A artist get no split and are not used.
- **FMA.** `fma_cross_dataset.build_split` uses `stable_int("fma-artist-split/<artist_id>") % 10` with the same 0-7 / 8 / 9 rule.
- Every runner aborts if an artist appears in two splits.
- **Label sets** use training tracks only:
  - Onion: the 50 most frequent canonicalised Last.fm tags with at least 400 training tracks. Only spelling and singular/plural variants are merged (`TAG_ALIASES`).
  - FMA: the 30 most frequent genres of `genres_all` with at least 200 training tracks.
  - Tracks without any selected label are dropped.
- **Preprocessing statistics** are fitted on training tracks only and applied unchanged to validation and test:
  - feature standardisation;
  - the reliability median;
  - the FMA TF-IDF vocabulary.

## Availability conditions and masks

| Paper | Onion / FMA code name | Six-view code name |
|---|---|---|
| Complete | `observed` | `observed` |
| −Audio / −Lyrics (FMA: −Text) / −Visual (FMA: −Social) | `no_audio`, `no_lyrics` / `no_text`, `no_visual` / `no_social` | `no_audio`, `no_lyrics`, `no_visual` (both views of the sense) |
| Random drop one | `random_one_missing` | `random_one_view_missing` |
| Random drop half | – | `random_half_views_missing` (removes ⌊k/2⌋ of k views) |
| Keep only one | `random_two_missing` | `keep_one_view` |

How the masks are generated and applied:

- The masks are produced by `intervention(...)` in `robust_multimodal_tagging.py`, `fma_cross_dataset.py` and `extended_modalities.py`.
- They depend only on the observed availability matrix of the test split and the seed, so every model of a dataset is evaluated on identical masks.
- A mask only removes modalities that were observed. Tracks with a single observed modality are left unchanged by the random conditions.
- AA-MCE maps every masked test track to its availability pattern.
- A track with no modality left uses the complete-pattern parameters. This happens under `no_lyrics` for a handful of Onion test tracks that originally lacked both audio and video.

## Validation-only selection

Test labels are never used to choose anything. The selection steps are:

1. **Training and thresholds of every model.**
   - Early stopping is on validation mAP: at most 30 epochs, patience 5, minimum improvement 1e-4.
   - Per-label decision thresholds maximise validation F1 on the grid 0.05:0.025:0.80 (31 values; ties go to the smallest threshold). They are computed on the best epoch's validation predictions.
2. **MCE.**
   - `optimized_multimodal_tagging --stage sweep` trains nine predeclared variants on seed 20260724 and never evaluates test data.
   - `select_tagging_ensemble` searches pairwise probability and rank averages of the five best variants, with weights 0.1-0.9, by validation mAP. It writes `ensemble_selection.json`: probability average, 0.6 × FullConcatMD10-H384-BCE + 0.4 × FullConcatMD-BCE.
   - `--stage final` refuses to run if the ensemble locked in the configuration differs from this selection.
3. **AA-MCE**, for each non-empty availability pattern π:
   - *Eligible tracks.* The eligible validation tracks are those whose observed modalities include π. Their availability is masked to π, and every candidate component predicts them under this mask.
   - *Weights.* Every weight vector of the simplex grid with step 0.1 is scored by validation mAP averaged over the five seeds: 66 vectors for three components, 1,001 for the five-candidate upper bound. The best vector replaces the locked weights (0.6, 0.4, 0) only if its gain exceeds δ = 0.0005 (`selection_margin`, a strict inequality).
   - *AA-MCE-2* is restricted to the two locked components.
   - *Thresholds.* Per-label thresholds are then selected per seed on the same masked validation tracks, from the mixed probabilities of each variant.
   - *Evaluation.* Test tracks use the weights and thresholds of their own pattern. Per-seed metrics use per-seed thresholds; the label-level tests use the five-seed mean predictions and mean thresholds.
4. **Platt calibration** (`posthoc_calibration.py`).
   - For every model, pattern and seed, `fit_platt` fits a per-label logistic regression on the logit of the probability, over the same masked validation tracks. A label without both classes keeps the identity.
   - `cal-obs` applies the complete-pattern parameters to every test track; `cal-AA` applies each track's pattern parameters.
   - Calibration metrics are computed on the five-seed mean of the calibrated probabilities.
   - A single Platt transform with a positive slope is increasing, so it leaves the ranking within its scope unchanged (`tests/test_calibration.py`).
   - Because `cal-AA` uses pattern-specific transforms and the metrics are computed on seed-averaged probabilities, the reported mAP moves only in the fourth decimal. For AA-MCE on complete observations it is 0.4533 raw, 0.4533 `cal-obs` and 0.4534 `cal-AA`. These are mAPs of the seed-mean probabilities, not the per-seed means of the main tables.
5. **Transfer to FMA and six views.** The recipe is not re-tuned: architectures, dropout rates, training settings and the 0.6/0.4 MCE weights are kept. The models are trained on the target dataset with early stopping on its validation artists. Only thresholds and, for AA-MCE, the pattern weights are selected on the target's validation artists. Per-pattern Platt calibration (step 4) was fitted on Onion only.

### Analysis history

The selection steps above never use test labels, but the study was not preregistered:

- AA-MCE was proposed after the fixed-weight MCE had been seen to fall behind RAMT and AttnFusion in keep-only-one Macro-F1 on the Onion test set.
- The candidate set, the weight grid and δ are recorded in the `resolved_config.yaml` of the first AA-MCE run (`availability_aware_ensemble_v1`). The FMA and six-view runs used them unchanged. The weak conditions of MCE in those two settings (FMA −Text, six-view keep-one-view) had, however, already been observed on test data.
- Per-pattern Platt calibration (`posthoc_calibration`), the δ-sensitivity analysis (`robustness_checks`) and `decision_layer_diagnostics` were added after the AA-MCE results, as post-hoc analyses.

Variants reported for AA-MCE (`aa_core.evaluate_variants`):

| Variant | Weights | Thresholds |
|---|---|---|
| MCE | locked 0.6 / 0.4 / 0 for every track | complete-pattern |
| MCE+AAT | locked | per pattern |
| AA-MCE-w | three-component weights selected for the complete pattern, used for every track | complete-pattern |
| AA-MCE-2 | per pattern, locked components only | per pattern |
| AA-MCE | per pattern, three candidates | per pattern |
| X / X+AAT | single baseline X (RAMT, AttnFusion, EvidFusion) | complete-pattern / per pattern |

## Statistics

- **Per-seed metrics.** mAP (macro AP), Macro-F1, Micro-F1 and macro ROC-AUC. Summaries report the mean, the standard deviation and 1.96·sd/√5 over the five seeds.
- **Label-level paired bootstrap** (`_bootstrap`). The per-label differences of AP or F1 are computed on the five-seed mean predictions with the seed-mean thresholds and resampled 5,000 times.
  - The two-sided p-value is 2·min(P(mean ≤ 0), P(mean ≥ 0)); the confidence interval is the 95 % percentile interval.
  - Holm correction is applied within each (proposed model, condition, metric) family. In `paper_data.py` the families are condition × metric per dataset.
- **Track-level bootstrap** (`robustness_checks.py`). 1,000 resamples of the test tracks for the macro-AP difference, with uncorrected p-values. The AP definition used there equals scikit-learn's in the absence of score ties; the run records the maximum discrepancy.
- **Calibration.** Macro ECE with 15 equal-width bins, Brier score and log loss.

## What is frozen

In the authors' workspace the following run directories were not modified after completion. Later runs read them, so they must be produced in the order of the README or the reproduction scripts.

| Run | Read by later runs as |
|---|---|
| `robust_multimodal_tagging_v3` | reference predictions for MCE and E2 tests; `labels.json` (label order, training tag frequencies) for the analyses and figures |
| `optimized_tagging_validation_v2` | `selection.json`, `ensemble_selection.json`, validation sweep and ensemble search (Fig. 3) |
| `optimized_multimodal_tagging_final_v1` | MCE predictions and thresholds; checkpoints of the components C1 and C2 reused by AA-MCE and by post-hoc calibration |
| `advanced_fusion_baselines_v1`, `full_feature_ablation_v1` | baseline predictions, thresholds, gate weights and efficiency tables |
| `fma_cross_dataset_v1`, `extended_modalities_v1` | frozen predictions compared with the AA-MCE transfer runs |
| `availability_aware_ensemble_v1` | checkpoints of RAMT, AttnFusion and EvidFusion and the pattern weights, reused by the five-candidate run and by post-hoc calibration |

The runners check label order and test-track order against these files and abort on any mismatch. The configuration fixes every hyper-parameter, seed, grid and margin. `results/paper_tables/` holds the aggregated tables used for the manuscript, so a reproduction can compare them with `outputs/paper/data`.

Not included in this repository:

- datasets;
- run outputs and model checkpoints (`outputs/`);
- the manuscript text and its typesetting code.

## Expected runtimes

Wall-clock times on the workstation above, as recorded in each run's `completion.json`:

| Run | Minutes |
|---|---:|
| robust_multimodal_tagging_v3 | 18.5 |
| optimized_tagging_validation_v2 (sweep) | 2.8 |
| optimized_multimodal_tagging_final_v1 | 4.0 |
| advanced_fusion_baselines_v1 (E2) | 9.0 |
| full_feature_ablation_v1 (E1-F) | 13.2 |
| fma_cross_dataset_v1 (E3) | 29.5 |
| extended_modalities_v1 (E7) | 31.7 |
| availability_aware_ensemble_v1 (E8) | 19.8 |
| availability_aware_ensemble_5c_v1 | 37.3 |
| availability_aware_ensemble_fma_v1 | 12.5 |
| availability_aware_ensemble_sixview_v1 | 31.6 |
| posthoc_calibration_v1 | 1.2 |
| robustness_checks_v1 (single-process CPU bootstrap) | 87.5 |
| paper_analysis_v1 | 0.5 |
| decision_layer_diagnostics_v1 (inference only) | 2.2 |

`select_tagging_ensemble`, `paper_efficiency` and the reporting modules take minutes at most. The experiments total about 5 h. Data preparation, which parses the bz2 feature files, is not included in this total. Run outputs occupy about 1.4 GB.
