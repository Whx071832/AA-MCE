# AA-MCE: availability-aware multi-capacity ensemble for music tagging under missing modalities

Code for the paper *Availability-Aware Social Music Tag Mining under Missing Modalities* (Chinese manuscript: 面向模态缺失的可用性感知社会化音乐标签挖掘方法; bibliographic details will be added after publication).

Multimodal taggers are usually trained and calibrated on fully observed tracks, but at test time some modalities are often missing, and the set of available modalities is known when the prediction is made. AA-MCE (availability-aware multi-capacity ensemble) uses this information only at the decision layer. Three component taggers are trained on training artists:

- **C1**: a concatenation tagger with hidden size 384 and modality dropout 0.1.
- **C2**: a concatenation tagger with hidden size 192 and modality dropout 0.3.
- **C3**: RAMT, a reliability-gated tagger with hidden size 192 and modality dropout 0.3.

For each of the non-empty modality-availability patterns (7 for three modalities), the following are selected on pattern-masked validation artists only:

- component weights, on a simplex grid with step 0.1;
- per-label decision thresholds, on a grid from 0.05 to 0.80 in steps of 0.025;
- optionally, per-label Platt calibration parameters.

The locked MCE weights (0.6, 0.4, 0) are replaced only when the validation mAP gain exceeds δ = 0.0005. At inference, each track's parameters are looked up in a per-pattern table. The fixed-weight MCE (0.6 × Concat-H384/MD10 + 0.4 × Concat-H192/MD30) is the ablation special case.

**Evaluation.** The main benchmark is Music4All-Onion with three views: Essentia audio (1,034-d), lyrics TF-IDF (1,000-d) and video ResNet (4,096-d).

- Split: artist-disjoint 8:1:1, assigned by BLAKE2b hashing of the artist id.
- Labels: the 50 most frequent Last.fm tags, counted on the training artists.
- Test conditions: six predeclared availability conditions, using the same fixed-seed masks for every model.

The method is transferred without re-tuning to two further settings:

- FMA, with audio, text and social views and 30 genres;
- a six-view Onion extension with 63 availability patterns.

In these transfers only the component weights and the thresholds are re-selected on the target's validation artists; per-pattern Platt calibration was fitted on Onion only.

All results use five seeds. Statistical tests are label-level paired bootstraps with Holm correction, complemented by a track-level bootstrap.

## Key results

Music4All-Onion, 50 tags, means over five seeds. "Missing mean" is the mean mAP over the five conditions with at least one modality removed. "Keep only one" retains one randomly chosen available modality per track.

| Model | Complete mAP | Missing mean mAP | Keep-only-one Macro-F1 |
|---|---:|---:|---:|
| **AA-MCE** | **0.4471** | **0.3892** | **0.3885** |
| MCE (fixed 0.6/0.4) | 0.4459 | 0.3868 | 0.3686 |
| Concat-H384/MD10 (C1) | 0.4390 | 0.3791 | 0.3614 |
| RAMT (C3) | 0.4286 | 0.3770 | 0.3790 |
| AttnFusion | 0.4326 | 0.3743 | 0.3732 |
| EvidFusion | 0.4220 | 0.3715 | 0.3421 |

Transfer without re-tuning, reported as missing-condition mean mAP:

- **FMA:** AA-MCE 0.4093 vs. MCE 0.4045.
- **Six-view Onion:** AA-MCE 0.4032 vs. MCE 0.3981.

Calibration: pattern-wise Platt calibration lowers AA-MCE's complete-observation macro ECE (15 bins) from 0.0921 to 0.0233.

All numbers are read from `results/paper_tables/`, and `tests/test_paper_tables.py` checks them.

## Repository layout

```
AA-MCE/
├── configs/experiments.yaml     all hyper-parameters, seeds and run-to-run dependencies
├── src/
│   ├── data/                    prepare.py (raw data -> Data/processed/v1), validate_processed.py
│   ├── experiments/             experiment runners: python -m src.experiments.<runner>
│   └── reporting/               paper tables and figures from outputs/experiments
├── scripts/                     reproduce_paper.ps1 / reproduce_paper.sh (whole pipeline)
├── results/paper_tables/        aggregated tables behind the paper's tables and figures
├── results/decision_layer_diagnostics/   outputs of decision_layer_diagnostics (Sec. 5.1 and 5.4 numbers)
├── tests/                       fast tests on synthetic data (no dataset needed)
├── docs/                        DATA.md, REPRODUCIBILITY.md, PAPER_MAP.md
├── Data/        (not tracked)   datasets, see docs/DATA.md
└── outputs/     (not tracked)   run directories, logs, regenerated tables and figures
```

Main modules in `src/experiments/`:

| Module | Role |
|---|---|
| `robust_multimodal_tagging.py` | Label set, targets, metrics, threshold search, availability masks; compact-feature reference family (`robust_multimodal_tagging_v3`) |
| `optimized_multimodal_tagging.py`, `select_tagging_ensemble.py` | Validation sweep, validation-only ensemble selection, locked MCE and its components C1/C2 |
| `full_feature_ablation.py`, `fusion_models.py` | Feature-matched baseline family incl. RAMT (C3); AttnFusion/EvidFusion/generic taggers |
| `advanced_fusion_baselines.py` | AttnFusion and EvidFusion baselines + efficiency audit |
| `aa_core.py`, `availability_aware_ensemble.py`, `availability_aware_campaigns.py` | AA-MCE: pattern enumeration, weight search with the δ rule, pattern thresholds, variants, tests (Onion, FMA, six views, five-candidate upper bound) |
| `fma_cross_dataset.py`, `extended_modalities.py` | FMA transfer and six-view study with the locked recipe |
| `posthoc_calibration.py`, `robustness_checks.py`, `paper_analysis.py`, `paper_efficiency.py` | Platt calibration, track-level bootstrap and margin sensitivity, calibration audit/tag groups/frozen-asset checks, latency and parameters |
| `decision_layer_diagnostics.py` | Post-hoc decision-layer diagnostics (no training): threshold shift and probability compression per availability pattern, decision-loss decomposition against an oracle-threshold upper bound, offline selection timing |
| `common.py` | Shared run utilities: config loading, seeding, device selection, environment record and the paired bootstrap `_bootstrap` |

`src/reporting/` contains four entry points and a style helper:

- `paper_data.py` rebuilds the tables shipped in `results/paper_tables/`, by default into `outputs/paper/data/`.
- `paper_facts.py` rebuilds `facts.json`.
- `paper_figures.py` draws the manuscript's Fig. 1-5.
- `final_figures.py` and `final_tables.py`, which share `paper_style.py`, produce an extended English figure and table suite for supplementary use.

## Installation

The code was used with Python 3.13.5, PyTorch 2.12.0+cu126 and Windows 11, and should work with Python ≥ 3.10. The code runs from the repository root and is not installed as a package.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # exact versions used for the paper
# alternatively: pip install ".[test]"  # minimum versions from pyproject.toml (dependencies only)
```

For GPU training, install the CUDA build of PyTorch that matches your driver, for example `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126`. The runners fall back to the CPU when CUDA is unavailable (`runtime.device: auto`).

The figure scripts need two fonts:

- `paper_figures.py` uses Times New Roman and SimSun, because it draws Chinese labels.
- `final_figures.py` stops with an error if Times New Roman is missing.

## Data

The datasets are **not redistributed** with this repository; obtain them from their providers under their own terms:

- **Music4All-Onion** (feature, tag and interaction files) is distributed on Zenodo.
- **Music4All** itself (audio, lyrics, metadata) must be requested from its authors. This code reads only the files listed below; the Onion and A+A files refer to Music4All tracks.
- **Music4All-A+A** provides artist- and album-level metadata with Onion track ids. It is used to assign tracks to artists for the artist-disjoint split.
- **MusicSem** `train.csv` is needed only because the `align` preparation stage, which writes the artist-disjoint split, also builds a MusicSem-Onion alignment. The tagging experiments do not use MusicSem data.
- **FMA** metadata (`fma_metadata.zip`) is available from https://github.com/mdeff/fma. The `fma_small` audio is optional; it only sets an `audio_available` flag.

Expected layout, derived from `src/data/prepare.py` and the FMA runners. The full stage-by-stage description is in [docs/DATA.md](docs/DATA.md).

```
Data/
├── music4all_onion/   id_essentia.tsv.bz2, id_lyrics_tf-idf.tsv.bz2, id_resnet.tsv.bz2, id_tags_dict.tsv.bz2,
│                      id_ivec1024.tsv.bz2, id_lyrics_word2vec.tsv.bz2, id_incp.tsv.bz2   (last three: six-view study)
├── music4all_aa/      artists_json/*.json, album_json/*.json, artist_modality_splits.json, album_modality_splits.json
├── musicsem/          train.csv
├── fma/fma_metadata/fma_metadata/   tracks.csv, genres.csv, features.csv, echonest.csv,
│                                    raw_tracks.csv, raw_albums.csv, raw_artists.csv
└── processed/v1/      written by python -m src.data.prepare
```

## Reproducing the paper

Run all commands from the repository root. Each experiment writes `outputs/experiments/<run-name>/` and finishes with a `completion.json`. The training runners refuse to write into a non-empty run directory unless `--overwrite` is given. The post-hoc runners (steps 7-8) overwrite their files.

The order matters because later runs read earlier ones:

- reference predictions and thresholds;
- component checkpoints;
- the validation-only ensemble selection.

`configs/experiments.yaml` names these dependencies.

```bash
# 0. data preparation (stages needed by this paper; `extract` and `interactions` are not needed)
python -m src.data.prepare --project-root . --stages aa musicsem fma features tags align manifest
python -m src.data.validate_processed --project-root .

# 1. compact-feature reference family: its predictions and labels.json are read by steps 3, 4, 7 and 8
python -m src.experiments.robust_multimodal_tagging --project-root . --run-name robust_multimodal_tagging_v3

# 2. validation sweep and validation-only ensemble selection (locks MCE = 0.6 x H384/MD10 + 0.4 x H192/MD30)
python -m src.experiments.optimized_multimodal_tagging --project-root . --run-name optimized_tagging_validation_v2 --stage sweep
python -m src.experiments.select_tagging_ensemble --project-root . --run-name optimized_tagging_validation_v2

# 3. locked MCE and its components C1/C2 (five seeds; checkpoints are reused by AA-MCE)
python -m src.experiments.optimized_multimodal_tagging --project-root . --run-name optimized_multimodal_tagging_final_v1 --stage final

# 4. baselines: AttnFusion/EvidFusion (+ efficiency audit) and the feature-matched family incl. RAMT
python -m src.experiments.advanced_fusion_baselines --project-root . --run-name advanced_fusion_baselines_v1
python -m src.experiments.full_feature_ablation --project-root . --run-name full_feature_ablation_v1

# 5. transfer of the locked recipe: FMA and six Onion views
python -m src.experiments.fma_cross_dataset --project-root . --run-name fma_cross_dataset_v1
python -m src.experiments.extended_modalities --project-root . --run-name extended_modalities_v1

# 6. AA-MCE: Onion, five-candidate upper bound, FMA, six views
python -m src.experiments.availability_aware_ensemble --project-root . --run-name availability_aware_ensemble_v1
python -m src.experiments.availability_aware_ensemble --project-root . --section availability_aware_ensemble_5c --run-name availability_aware_ensemble_5c_v1
python -m src.experiments.availability_aware_campaigns --project-root . --dataset fma --run-name availability_aware_ensemble_fma_v1
python -m src.experiments.availability_aware_campaigns --project-root . --dataset sixview --run-name availability_aware_ensemble_sixview_v1

# 7. post-hoc analyses (no training; post-hoc calibration runs inference with the saved checkpoints)
python -m src.experiments.posthoc_calibration --project-root . --run-name posthoc_calibration_v1
python -m src.experiments.robustness_checks --project-root . --run-name robustness_checks_v1
python -m src.experiments.paper_analysis --project-root . --run-name paper_analysis_v1
python -m src.experiments.paper_efficiency --project-root .        # writes outputs/experiments/paper_efficiency_v1
python -m src.experiments.decision_layer_diagnostics --project-root . --run-name decision_layer_diagnostics_v1   # Sec. 5.1/5.4 diagnostics

# 8. tables and figures (paths are relative to the repository root)
python -m src.reporting.paper_data --out outputs/paper/data         # same files as results/paper_tables/ except facts.json
python -m src.reporting.paper_facts                                  # outputs/paper/data/facts.json
python -m src.reporting.paper_figures --data outputs/paper/data --out outputs/paper/figures
python -m src.reporting.final_figures --project-root .              # extended suite: outputs/figures_final
python -m src.reporting.final_tables --project-root .               # extended suite: outputs/tables_final
```

The same sequence is automated by the reproduction scripts. They skip every stage whose `completion.json` already exists; the ensemble-selection step checks for `ensemble_selection.json` instead. A failed stage is retried once, and logs go to `outputs/logs/`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_paper.ps1 [-DryRun] [-SkipDataPrep] [-Python <path>]
```
```bash
bash scripts/reproduce_paper.sh [--dry-run] [--skip-data-prep]      # PYTHON=/path/to/python to override
```

Figures 1, 2 and 4 of the manuscript can be redrawn from the shipped tables without any data:

```bash
python -m src.reporting.paper_figures --data results/paper_tables --out outputs/paper/figures --only 1 2 4
```

Figures 3 and 5 also read small CSV files from `outputs/experiments/`.

## Paper-to-code map

| Paper item | Runs | Code | Tables / files |
|---|---|---|---|
| Table 1 (performance and cost, Onion) | optimized_multimodal_tagging_final_v1, full_feature_ablation_v1, advanced_fusion_baselines_v1, availability_aware_ensemble_v1, paper_efficiency_v1 | `paper_data.py`, `paper_efficiency.py` | `onion_summary.csv`, `onion_tests.csv`, `efficiency.csv` |
| Table 2 (AA-MCE ablation and controls) | availability_aware_ensemble_v1, availability_aware_ensemble_5c_v1 | `availability_aware_ensemble.py`, `paper_data.py` | `ablation_summary.csv`, `aa_variant_tests.csv`, `aa5_tests.csv` |
| Table 3 (FMA and six-view transfer) | fma_cross_dataset_v1, extended_modalities_v1, availability_aware_ensemble_{fma,sixview}_v1 | `availability_aware_campaigns.py`, `paper_data.py` | `fma_*.csv`, `sixview_*.csv` |
| Fig. 1-5 | see [docs/PAPER_MAP.md](docs/PAPER_MAP.md) | `paper_figures.py` | `outputs/paper/figures/Fig*.pdf/.png` |

[docs/PAPER_MAP.md](docs/PAPER_MAP.md) maps every table, figure and section to its runs, modules and output files. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) describes seeds, split hashing, masks, validation-only selection and what is frozen.

## Hardware and runtime

All runs used one NVIDIA RTX 3090 (24 GB), an AMD Ryzen 5 7500F and 32 GB RAM. The table gives wall-clock times as recorded in each run's `completion.json`.

| Run | Minutes | Run | Minutes |
|---|---:|---|---:|
| robust_multimodal_tagging_v3 | 18.5 | extended_modalities_v1 (E7) | 31.7 |
| optimized_tagging_validation_v2 (sweep) | 2.8 | availability_aware_ensemble_v1 (E8) | 19.8 |
| optimized_multimodal_tagging_final_v1 | 4.0 | availability_aware_ensemble_5c_v1 | 37.3 |
| advanced_fusion_baselines_v1 (E2) | 9.0 | availability_aware_ensemble_fma_v1 | 12.5 |
| full_feature_ablation_v1 (E1-F) | 13.2 | availability_aware_ensemble_sixview_v1 | 31.6 |
| fma_cross_dataset_v1 (E3) | 29.5 | posthoc_calibration_v1 | 1.2 |
| paper_analysis_v1 | 0.5 | robustness_checks_v1 (CPU bootstrap) | 87.5 |
| decision_layer_diagnostics_v1 | 2.2 | | |

The experiments take about 5 h in total, plus data preparation. The disk space needed is about 11 GB:

- about 5 GB of raw inputs;
- about 4.5 GB in `Data/processed/v1`;
- about 1.4 GB of run outputs.

## Analysis history

The selection procedures never use test labels, but the study was not preregistered, and the manuscript states this:

- AA-MCE was proposed after the fixed-weight MCE was seen to fall behind RAMT and AttnFusion in keep-only-one Macro-F1 on the Onion test set.
- Its candidate set, weight grid and δ were fixed before the first AA-MCE run and were then used unchanged on FMA and six views.
- Per-pattern Platt calibration, the δ-sensitivity analysis and `decision_layer_diagnostics` were added afterwards, as post-hoc analyses.

## Tests

```bash
python -m pytest
```

The tests run in a few seconds on synthetic data and need no dataset. They cover:

- availability-pattern enumeration (7 patterns for three modalities, 63 for six);
- the simplex weight grid and the δ replacement rule;
- per-label threshold selection;
- Platt calibration monotonicity and ECE;
- deterministic BLAKE2b artist splits and the fixed-seed availability masks;
- the paired bootstrap and Holm correction;
- component forward passes;
- the per-pattern AA-MCE lookup;
- consistency of the configuration and of the key results above with `results/paper_tables/`.

## Citation

```bibtex
@article{aamce2026,
  title   = {Availability-Aware Social Music Tag Mining under Missing Modalities},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  note    = {TODO: complete after publication}
}
```

See also `CITATION.cff`.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are not covered by this license.

## 中文说明

本仓库是论文《面向模态缺失的可用性感知社会化音乐标签挖掘方法》的配套代码，提供 AA-MCE（可用性感知多容量集成）的全部实验与作图代码。

AA-MCE 由三个组件网络组成：

- C1：拼接网络，隐层 384，模态丢弃率 0.1；
- C2：拼接网络，隐层 192，模态丢弃率 0.3；
- C3：可靠性门控网络 RAMT，隐层 192，模态丢弃率 0.3。

对每一种模态可用模式，只在按该模式掩码的验证集艺术家上选择三类参数：

- 组件权重：单纯形网格，步长 0.1；锁定权重 (0.6, 0.4, 0) 仅在验证增益超过 δ=0.0005 时被替换；
- 逐标签决策阈值；
- Platt 校准参数。

推理时按样本的可用模式查表。固定权重的 MCE 是其消融特例。

- 数据：Music4All-Onion（Zenodo）、Music4All（需向原作者申请）、Music4All-A+A、MusicSem 与 FMA（https://github.com/mdeff/fma）均**不随仓库分发**，请按 `docs/DATA.md` 的目录结构放入 `Data/`。其中 MusicSem 仅因 `align` 数据阶段（生成艺术家隔离划分）需要读取而必须存在。
- 运行：在仓库根目录依次执行上文“Reproducing the paper”中的命令，或直接运行 `scripts\reproduce_paper.ps1`（Linux 下为 `scripts/reproduce_paper.sh`）。已存在 `completion.json` 的阶段会被自动跳过。
- 迁移：FMA 与六视图场景只在各自验证集上重新选择组件权重与阈值；按模式 Platt 校准只在 Onion 上实施。
- 研究过程说明：AA-MCE 是在观察到固定权重集成 MCE 在 Onion 测试集“仅留一”条件下 Macro-F1 落后于 RAMT 与 AttnFusion 之后提出的；其候选组件、权重网格与 δ 在首次运行前确定并原样用于迁移场景；按模式 Platt 校准、δ 敏感性分析与 `decision_layer_diagnostics`（阈值偏移、概率压缩、判决损失分解与离线选择耗时）均为事后分析。
- 输出：实验结果写入 `outputs/experiments/<运行名>/`，论文表格数据写入 `outputs/paper/data`，与 `results/paper_tables/` 中随仓库提供的汇总表对应；论文图 1-5 写入 `outputs/paper/figures`（需要 Times New Roman 与宋体 SimSun 字体）。
- 耗时：单张 RTX 3090 上全部实验约 5 小时（不含数据预处理），磁盘约需 11 GB。
- 测试：`python -m pytest` 在合成数据上数秒内完成，无需任何数据集。
