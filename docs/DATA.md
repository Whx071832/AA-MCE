# Data

This repository contains **no dataset files**, and none of the datasets below is redistributed. Obtain them from their providers and follow each provider's license and terms of use. Place them under `Data/` in the repository root. `Data/` is ignored by git.

## Sources

| Dataset | What this code uses | Where to obtain |
|---|---|---|
| Music4All-Onion | Track-level features of three senses and the Last.fm tags with weights | Zenodo (the Music4All-Onion release). TODO (authors): add the DOI of the version used |
| Music4All | Not read directly. The Onion and A+A track ids refer to Music4All tracks | Must be requested from the Music4All authors (Santana et al., 2020) |
| Music4All-A+A | Artist and album metadata with the Onion track ids of every artist. It defines the artist of each track and hence the artist-disjoint split | TODO (authors): add the official download link |
| MusicSem | `train.csv` only. Required because the `align` stage, which also writes the artist-disjoint split, builds a MusicSem-Onion alignment first. No MusicSem data enters the tagging experiments | Dataset card: MusicSem (Salganik et al.), MIT license. TODO (authors): add the link |
| FMA | `fma_metadata.zip`: track metadata, genres, librosa features, Echonest descriptors, raw engagement statistics | https://github.com/mdeff/fma (the `fma_small` audio is optional) |

The FMA audio files are licensed individually by their artists; see the FMA repository for the licensing of the metadata.

## Expected layout of `Data/`

The layout is derived from `src/data/prepare.py` and the FMA runners. File names must match exactly.

```
Data/
├── music4all_onion/
│   ├── id_essentia.tsv.bz2            audio, Essentia descriptors, 1,034-d      (three views + six views)
│   ├── id_lyrics_tf-idf.tsv.bz2       lyrics, TF-IDF, 1,000-d                   (three views + six views)
│   ├── id_resnet.tsv.bz2              video, ResNet frame max+mean, 4,096-d     (three views + six views)
│   ├── id_ivec1024.tsv.bz2            audio, MFCC i-vector (1,024 GMM), 400-d   (six views only)
│   ├── id_lyrics_word2vec.tsv.bz2     lyrics, averaged word2vec, 300-d          (six views only)
│   ├── id_incp.tsv.bz2                video, Inception frame max+mean, 4,096-d  (six views only)
│   ├── id_tags_dict.tsv.bz2           Last.fm tags and weights per track
│   ├── processed_lyrics.tar.gz        optional: only for the unused `extract` stage
│   └── userid_trackid_timestamp.tsv.bz2  optional: only for the unused `interactions` stage
├── music4all_aa/
│   ├── artists_json/*.json            one file per artist ("artist_info", "music4all_onion_id", ...)
│   ├── album_json/*.json              one file per album
│   ├── artist_modality_splits.json
│   └── album_modality_splits.json
├── musicsem/
│   └── train.csv
├── fma/
│   ├── fma_metadata/fma_metadata/     contents of fma_metadata.zip
│   │   ├── tracks.csv, genres.csv                                   read by the `fma` stage
│   │   └── features.csv, echonest.csv, raw_tracks.csv, raw_albums.csv, raw_artists.csv
│   │                                                                read directly by the FMA runners
│   └── fma_small/fma_small/           optional: <NNN>/<NNNNNN>.mp3 (track id), only sets `audio_available`
└── processed/v1/                      written by `python -m src.data.prepare`
```

The copy used for the paper had the following contents:

- Music4All-Onion features for 109,269 tracks; the two video views cover 98,877 of them.
- 6,741 artist and 19,511 album JSON files for Music4All-A+A.
- 35,977 MusicSem rows.

## Preparation stages

`python -m src.data.prepare --project-root . --stages <stage> [...]` runs the requested stages. They always execute in the fixed order below, whatever the order on the command line. Outputs that already exist are skipped unless `--overwrite` is given. The pipeline never deserialises pickle files and never downloads audio.

| Stage | Reads | Writes under `Data/processed/v1/` | Needed here |
|---|---|---|---|
| `extract` | `music4all_onion/processed_lyrics.tar.gz` | nothing (extracts next to the archive) | no |
| `aa` | `music4all_aa/*` | `entities/aa_artists.parquet`, `aa_albums.parquet`, `aa_track_artists.parquet`, `aa_track_albums.parquet`, modality-condition tables, `aa_parse_errors.json` | **yes** |
| `musicsem` | `musicsem/train.csv` | `social/musicsem_social_semantics.parquet`, `musicsem_retrieval_queries.parquet`, `musicsem_thread_splits.json` | **yes** (read by `align`) |
| `fma` | `fma/fma_metadata/fma_metadata/tracks.csv`, `genres.csv` | `external/fma_tracks.parquet`, `fma_artists.parquet`, `fma_albums.parquet`, `fma_genres.parquet` | **yes** |
| `features` | the six `music4all_onion/id_*.tsv.bz2` feature files | `features/onion/<view>/part-*.npy`, `part-*.track_ids.npy`, `index.parquet`, `metadata.json` | **yes** (all six views) |
| `tags` | `music4all_onion/id_tags_dict.tsv.bz2`, the three-view feature indexes, the `aa` edges | `features/onion/tag_weights/*`, `entities/onion_tracks.parquet` | **yes** |
| `align` | outputs of `musicsem`, `aa` and `tags` | `alignments/*`, `entities/aa_artist_cold_start_splits.parquet`, `entities/onion_track_artist_disjoint_splits.parquet` | **yes** (the artist-disjoint split) |
| `interactions` | `music4all_onion/userid_trackid_timestamp.tsv.bz2` (252,984,396 events) | `interactions/*` (temporal split of the earlier recommendation study) | no |
| `manifest` | processed outputs and source files | `data_manifest.json` | bookkeeping |

The command for this paper is:

```bash
python -m src.data.prepare --project-root . --stages aa musicsem fma features tags align manifest
python -m src.data.validate_processed --project-root .
```

Do not use `--stages all`: it includes `interactions`, which processes the full listening log and is not needed here.

`validate_processed` checks the following and exits with status 1 on any failure:

- the required tables exist;
- Onion track ids are unique;
- MusicSem thread splits are consistent;
- feature indexes contain no duplicate ids;
- the MusicSem-Onion alignment is consistent;
- the track split agrees with the artist split.

Its checks of the temporal interaction split run only if the `interactions` stage was run.

## What the experiment runners read

- **Onion runners** read three kinds of processed files:
  - `entities/onion_tracks.parquet` (`track_id`, `tags`);
  - `entities/onion_track_artist_disjoint_splits.parquet` (`track_id`, `artist_id`, `cold_start_split`);
  - `features/onion/<view>/`.

  The three-view study uses `audio_essentia`, `lyrics_tfidf` and `visual_resnet`. The six-view study adds `audio_ivec1024`, `lyrics_word2vec` and `visual_incp`.
- **FMA runners** (`fma_cross_dataset`, `availability_aware_campaigns --dataset fma`, `paper_analysis`, `robustness_checks`) read two processed tables, `external/fma_tracks.parquet` and `external/fma_genres.parquet`. They also read five raw CSVs from `Data/fma/fma_metadata/fma_metadata/`: `features.csv`, `raw_artists.csv`, `raw_tracks.csv`, `raw_albums.csv` and `echonest.csv`.
- Preprocessing statistics are fitted on training tracks only:
  - feature standardisation;
  - the per-view reliability median;
  - the FMA TF-IDF vocabulary (2,000 terms, `min_df=3`).

## Resulting task sizes

Taken from `results/paper_tables/numbers.json`.

| Dataset | Train tracks / artists | Validation tracks / artists | Test tracks / artists | Labels |
|---|---|---|---|---|
| Music4All-Onion | 46,232 / 4,829 | 4,934 / 584 | 5,702 / 585 | 50 Last.fm tags (training frequency 4,594-25,318) |
| FMA | 83,188 / 12,021 | 8,912 / 1,505 | 10,656 / 1,564 | 30 genres |

Modality coverage of the Onion test split is 99.9 % audio, 100 % lyrics and 90.0 % video. The FMA Echonest social descriptors exist for 12.0 % of FMA test tracks.

## Disk space

Approximate sizes for the paper's configuration:

| Item | Size |
|---|---|
| Onion input files | about 3.2 GB |
| Extracted FMA metadata | about 1.4 GB |
| `Data/processed/v1` without the unused interaction split | about 4.5 GB, of which the six feature views are 4.3 GB |
| Run outputs in `outputs/experiments/` | about 1.4 GB |
