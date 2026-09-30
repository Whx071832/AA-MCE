"""E3: external cross-dataset validation on FMA multi-label genre tagging.

The method recipe locked on Music4All-Onion (architectures, modality dropout
rates, the multi-capacity ensemble components and their 0.6/0.4 weights) is
transferred to FMA without re-tuning.  Only per-label decision thresholds are
calibrated on the FMA validation artists.

Task: multi-label genre tagging over the top training genres.  Modalities:
- audio: the 518 librosa descriptors shipped with FMA;
- text: TF-IDF over track title, album title, artist name, track tags and the
  artist biography (vocabulary fitted on training artists only);
- social: engagement statistics (track/album/artist listens, favourites,
  interest, comments) plus the five Echonest social descriptors where
  available (naturally missing for most tracks).

Splits are deterministic artist-disjoint 80/10/10 by artist hash; any artist
overlap aborts the run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from .common import _bootstrap
from .fusion_models import AttnFusionTagger, EvidFusionTagger, GenericTagger, evidential_loss
from .common import environment_record, load_config, resolve_device, set_seed
from .robust_multimodal_tagging import holm_by_condition, metric_values, stable_int, summarise, tune_thresholds


MODALITIES = ("audio", "text", "social")
CONDITIONS = ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"]
TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return TAG_PATTERN.sub(" ", str(value)).strip()


def build_split(artist_id: str) -> str:
    bucket = stable_int(f"fma-artist-split/{artist_id}") % 10
    if bucket <= 7:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def build_task(processed_root: Path, fma_root: Path, settings: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    tracks = pd.read_parquet(
        processed_root / "external" / "fma_tracks.parquet",
        columns=["track_id", "source_track_id", "title", "artist_id", "artist_name", "album_id", "album_title", "genres_all", "tags"],
    )
    genres = pd.read_parquet(processed_root / "external" / "fma_genres.parquet")
    genre_names = {str(row.genre_id): str(row.title) for row in genres.itertuples()}
    tracks["genre_titles"] = tracks["genres_all"].map(
        lambda values: sorted({genre_names[str(v)] for v in (values.tolist() if isinstance(values, np.ndarray) else list(values or [])) if str(v) in genre_names})
    )
    tracks = tracks.loc[tracks["genre_titles"].map(bool)].copy()
    tracks["split"] = tracks["artist_id"].astype(str).map(build_split)

    train_mask = tracks["split"] == "train"
    frequencies: dict[str, int] = {}
    for titles in tracks.loc[train_mask, "genre_titles"]:
        for title in titles:
            frequencies[title] = frequencies.get(title, 0) + 1
    labels = [
        title
        for title, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= int(settings["min_train_label_frequency"])
    ][: int(settings["top_labels"])]
    index = {label: position for position, label in enumerate(labels)}
    tracks["label_ids"] = tracks["genre_titles"].map(lambda titles: [index[t] for t in titles if t in index])
    tracks = tracks.loc[tracks["label_ids"].map(bool)].copy()

    # --- audio modality -----------------------------------------------------
    features = pd.read_csv(fma_root / "features.csv", header=[0, 1, 2], index_col=0)
    features.index = features.index.astype(np.int64)
    audio_lookup = {int(track): row for track, row in zip(features.index, range(len(features)))}
    audio_matrix = features.to_numpy(dtype=np.float32)
    del features

    # --- text modality ------------------------------------------------------
    raw_artists = pd.read_csv(
        fma_root / "raw_artists.csv",
        usecols=["artist_id", "artist_bio", "tags"],
        dtype={"artist_id": str},
    ).set_index("artist_id")
    bio = raw_artists["artist_bio"].map(clean_text)
    artist_tags = raw_artists["tags"].map(clean_text)

    def corpus(row: Any) -> str:
        tags = row.tags.tolist() if isinstance(row.tags, np.ndarray) else list(row.tags or [])
        artist_key = str(row.artist_id)
        pieces = [
            clean_text(row.title),
            clean_text(row.album_title),
            clean_text(row.artist_name),
            " ".join(str(tag) for tag in tags),
            str(bio.get(artist_key, "")),
            str(artist_tags.get(artist_key, "")),
        ]
        return " ".join(piece for piece in pieces if piece)

    tracks["text_corpus"] = [corpus(row) for row in tracks.itertuples()]

    # --- social modality ----------------------------------------------------
    raw_tracks = pd.read_csv(
        fma_root / "raw_tracks.csv",
        usecols=["track_id", "album_id", "artist_id", "track_listens", "track_favorites", "track_interest", "track_comments"],
    )
    raw_tracks["track_id"] = raw_tracks["track_id"].astype(np.int64)
    raw_albums = pd.read_csv(fma_root / "raw_albums.csv", usecols=["album_id", "album_listens", "album_favorites"])
    raw_artist_counts = pd.read_csv(fma_root / "raw_artists.csv", usecols=["artist_id", "artist_favorites", "artist_comments"])
    social = raw_tracks.merge(raw_albums, on="album_id", how="left").merge(raw_artist_counts, on="artist_id", how="left")
    social = social.set_index("track_id")
    count_columns = [
        "track_listens",
        "track_favorites",
        "track_interest",
        "track_comments",
        "album_listens",
        "album_favorites",
        "artist_favorites",
        "artist_comments",
    ]
    for column in count_columns:
        social[column] = np.log1p(pd.to_numeric(social[column], errors="coerce").fillna(0.0).clip(lower=0.0))
    echonest = pd.read_csv(fma_root / "echonest.csv", header=[0, 1, 2], index_col=0)
    echonest.index = echonest.index.astype(np.int64)
    social_block = echonest.loc[:, echonest.columns.get_level_values(1) == "social_features"]
    social_block.columns = [f"echonest_{name}" for name in social_block.columns.get_level_values(2)]
    social = social.join(social_block, how="left")
    social["echonest_available"] = social["echonest_artist_familiarity"].notna().astype(np.float32)
    echonest_columns = [column for column in social.columns if column.startswith("echonest_") and column != "echonest_available"]
    for column in echonest_columns:
        social[column] = pd.to_numeric(social[column], errors="coerce").fillna(0.0)
    social_columns = count_columns + echonest_columns + ["echonest_available"]
    social_matrix_full = social[social_columns].to_numpy(dtype=np.float32)
    social_lookup = {int(track): row for track, row in zip(social.index, range(len(social)))}

    # --- assemble per-track matrices ---------------------------------------
    source_ids = tracks["source_track_id"].astype(np.int64).to_numpy()
    n = len(tracks)
    audio = np.zeros((n, audio_matrix.shape[1]), dtype=np.float32)
    audio_present = np.zeros(n, dtype=bool)
    social_values = np.zeros((n, len(social_columns)), dtype=np.float32)
    social_present = np.zeros(n, dtype=bool)
    for position, source_id in enumerate(source_ids):
        row = audio_lookup.get(int(source_id))
        if row is not None:
            audio[position] = audio_matrix[row]
            audio_present[position] = True
        row = social_lookup.get(int(source_id))
        if row is not None:
            social_values[position] = social_matrix_full[row]
            social_present[position] = True
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    split_frames = {
        split: tracks.loc[tracks["split"] == split].sort_values("track_id", kind="stable").reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    artists = {split: set(frame["artist_id"].astype(str)) for split, frame in split_frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = artists[left] & artists[right]
        if overlap:
            raise RuntimeError(f"Artist leakage between {left} and {right}: {len(overlap)}")

    # Rebuild row order aligned with concatenated split frames.
    position_of_track = {track: position for position, track in enumerate(tracks["track_id"].to_numpy())}
    aligned_rows = np.concatenate(
        [np.asarray([position_of_track[track] for track in split_frames[split]["track_id"].to_numpy()], dtype=np.int64) for split in ("train", "validation", "test")]
    )
    offsets: dict[str, np.ndarray] = {}
    start = 0
    for split in ("train", "validation", "test"):
        offsets[split] = np.arange(start, start + len(split_frames[split]), dtype=np.int64)
        start += len(split_frames[split])

    audio = audio[aligned_rows]
    audio_present = audio_present[aligned_rows]
    social_values = social_values[aligned_rows]
    social_present = social_present[aligned_rows]
    corpus_series = tracks["text_corpus"].to_numpy()[aligned_rows]

    # TF-IDF fitted on training artists only.
    vectorizer = TfidfVectorizer(max_features=int(settings["text_max_features"]), norm=None, sublinear_tf=True, min_df=3)
    train_rows = offsets["train"]
    vectorizer.fit([corpus_series[row] for row in train_rows])
    text = np.asarray(vectorizer.transform(corpus_series).todense(), dtype=np.float32)
    text_present = np.asarray([len(document.strip()) > 0 for document in corpus_series], dtype=bool)

    def finalise(matrix: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_quality = np.linalg.norm(matrix, axis=1)
        train_present = train_rows[present[train_rows]]
        mean = matrix[train_present].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = matrix[train_present].std(axis=0, dtype=np.float64).astype(np.float32)
        matrix -= mean
        matrix /= np.maximum(std, 1e-6)
        matrix[~present] = 0.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-6)
        median_quality = float(np.median(raw_quality[train_present]))
        reliability = np.zeros(len(matrix), dtype=np.float32)
        reliability[present] = np.clip(raw_quality[present] / max(median_quality, 1e-6), 0.2, 2.0)
        return matrix, reliability

    audio, audio_reliability = finalise(audio, audio_present)
    text, text_reliability = finalise(text, text_present)
    social_values, social_reliability = finalise(social_values, social_present)

    features_by_modality = {"audio": audio, "text": text, "social": social_values}
    availability = np.stack([audio_present, text_present, social_present], axis=1).astype(np.float32)
    reliability = np.stack([audio_reliability, text_reliability, social_reliability], axis=1)
    split_frames = {split: frame for split, frame in split_frames.items()}
    for split, rows in offsets.items():
        split_frames[split] = split_frames[split].assign(_row=rows)
    return split_frames, labels, features_by_modality, availability, reliability


def targets_for(frame: pd.DataFrame, labels: int) -> np.ndarray:
    values = np.zeros((len(frame), labels), dtype=np.float32)
    for row, label_ids in enumerate(frame["label_ids"]):
        values[row, label_ids] = 1.0
    return values


def build_model(name: str, input_dimensions: list[int], labels: int, settings: dict[str, Any]) -> nn.Module:
    hidden = int(settings["hidden_dimension"])
    dropout = float(settings["representation_dropout"])
    md = float(settings["modality_dropout"])
    new_md = float(settings["new_model_modality_dropout"])
    generic = {
        "AudioOnly": {"fusion": "mean", "active_modalities": (0,)},
        "TextOnly": {"fusion": "mean", "active_modalities": (1,)},
        "SocialOnly": {"fusion": "mean", "active_modalities": (2,)},
        "UniformMean": {"fusion": "mean"},
        "EarlyConcat": {"fusion": "concat"},
        "ConcatModDrop": {"fusion": "concat", "modality_dropout": md},
        "ReliabilityGate": {"fusion": "reliability"},
        "RAMT": {"fusion": "reliability", "modality_dropout": md},
        "ConcatMD10-H384": {"fusion": "concat", "modality_dropout": float(settings["ensemble_component_dropout"])},
    }
    if name in generic:
        spec = generic[name]
        return GenericTagger(
            input_dimensions,
            hidden_dimension=int(settings["ensemble_hidden_dimension"]) if name == "ConcatMD10-H384" else hidden,
            labels=labels,
            fusion=str(spec["fusion"]),
            active_modalities=spec.get("active_modalities"),
            modality_dropout=float(spec.get("modality_dropout", 0.0)),
            representation_dropout=dropout,
        )
    if name == "AttnFusion":
        return AttnFusionTagger(
            input_dimensions,
            hidden_dimension=hidden,
            labels=labels,
            heads=int(settings["attention_heads"]),
            layers=int(settings["attention_layers"]),
            modality_dropout=new_md,
            representation_dropout=dropout,
        )
    if name == "EvidFusion":
        return EvidFusionTagger(
            input_dimensions,
            hidden_dimension=hidden,
            labels=labels,
            modality_dropout=new_md,
            representation_dropout=dropout,
        )
    raise ValueError(name)


@torch.no_grad()
def predict(
    model: nn.Module,
    features: list[np.ndarray],
    availability: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    for start in range(0, len(availability), batch_size):
        stop = min(start + batch_size, len(availability))
        logits, _ = model(
            [torch.as_tensor(values[start:stop], device=device) for values in features],
            torch.as_tensor(availability[start:stop], device=device),
            torch.as_tensor(reliability[start:stop], device=device),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_model(
    name: str,
    seed: int,
    features: dict[str, list[np.ndarray]],
    availability: dict[str, np.ndarray],
    reliability: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, pd.DataFrame]:
    set_seed(seed, deterministic=True)
    model = build_model(name, [values.shape[1] for values in features["train"]], target["train"].shape[1], settings).to(device)
    ratio = (len(target["train"]) - target["train"].sum(axis=0)) / np.maximum(target["train"].sum(axis=0), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(np.sqrt(np.clip(ratio, 1.0, 25.0)), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_map = -math.inf
    best_thresholds = np.full(target["train"].shape[1], 0.5, dtype=np.float32)
    stale = 0
    history_rows = []
    annealing_epochs = float(settings["evidence_annealing_epochs"])
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        order = rng.permutation(len(target["train"]))
        for start in range(0, len(order), int(settings["batch_size"])):
            selected = order[start : start + int(settings["batch_size"])]
            batch_features = [torch.as_tensor(values[selected], device=device) for values in features["train"]]
            batch_available = torch.as_tensor(availability["train"][selected], device=device)
            batch_reliability = torch.as_tensor(reliability["train"][selected], device=device)
            batch_target = torch.as_tensor(target["train"][selected], device=device)
            optimizer.zero_grad(set_to_none=True)
            if name == "EvidFusion":
                combined, per_modality = model.opinions(batch_features, batch_available)
                loss = evidential_loss(combined, per_modality, batch_available, batch_target, annealing=min(1.0, epoch / annealing_epochs))
            else:
                logits, _ = model(batch_features, batch_available, batch_reliability)
                loss = criterion(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability = predict(
            model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"])
        )
        validation_map = float(average_precision_score(target["validation"], validation_probability, average="macro"))
        history_rows.append({"model": name, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mAP": validation_map})
        if validation_map > best_map + float(settings["minimum_improvement"]):
            best_map = validation_map
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_thresholds = tune_thresholds(target["validation"], validation_probability)
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError(name)
    model.load_state_dict(best_state)
    return model, best_thresholds, pd.DataFrame(history_rows)


def intervention(name: str, availability: np.ndarray, seed: int) -> np.ndarray:
    result = availability.copy()
    if name == "observed":
        return result
    if name.startswith("no_"):
        result[:, MODALITIES.index(name[3:])] = 0.0
        return result
    rng = np.random.default_rng(seed)
    if name == "random_one_missing":
        for row in range(len(result)):
            candidates = np.flatnonzero(result[row] > 0)
            if len(candidates) > 1:
                result[row, rng.choice(candidates)] = 0.0
        return result
    if name == "random_two_missing":
        for row in range(len(result)):
            candidates = np.flatnonzero(result[row] > 0)
            if len(candidates) > 1:
                keep = int(rng.choice(candidates))
                result[row] = 0.0
                result[row, keep] = 1.0
        return result
    raise ValueError(name)


def paired_tests(
    target: np.ndarray,
    labels: list[str],
    mean_probabilities: dict[str, dict[str, np.ndarray]],
    mean_thresholds: dict[str, np.ndarray],
    proposed: str,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_label_rows, test_rows = [], []
    for condition in CONDITIONS:
        ap_cache: dict[str, np.ndarray] = {}
        f1_cache: dict[str, np.ndarray] = {}
        for model, condition_values in mean_probabilities.items():
            probability = condition_values[condition]
            predicted = probability >= mean_thresholds[model][None, :]
            ap_cache[model] = np.asarray([average_precision_score(target[:, label], probability[:, label]) for label in range(len(labels))])
            f1_cache[model] = np.asarray([f1_score(target[:, label], predicted[:, label], zero_division=0) for label in range(len(labels))])
            for label_id, label in enumerate(labels):
                per_label_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "label_id": label_id,
                        "label": label,
                        "AP": float(ap_cache[model][label_id]),
                        "F1": float(f1_cache[model][label_id]),
                    }
                )
        for baseline in mean_probabilities:
            if baseline == proposed:
                continue
            for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                delta = cache[proposed] - cache[baseline]
                low, high, p_value = _bootstrap(delta, 20260815 + len(test_rows), replicates)
                test_rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "proposed": proposed,
                        "baseline": baseline,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(per_label_rows), holm_by_condition(pd.DataFrame(test_rows))


def run(project_root: Path, config: dict[str, Any], run_name: str, overwrite: bool) -> Path:
    settings = config["fma_cross_dataset"]
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = resolve_device(config["runtime"]["device"])
    processed_root = project_root / config["project"]["processed_root"]
    fma_root = project_root / "Data" / "fma" / "fma_metadata" / "fma_metadata"
    split_frames, labels, features_by_modality, availability_all, reliability_all = build_task(processed_root, fma_root, settings)

    features = {
        split: [features_by_modality[m][frame["_row"].to_numpy()] for m in MODALITIES] for split, frame in split_frames.items()
    }
    availability = {split: availability_all[frame["_row"].to_numpy()] for split, frame in split_frames.items()}
    reliability = {split: reliability_all[frame["_row"].to_numpy()] for split, frame in split_frames.items()}
    target = {split: targets_for(frame, len(labels)) for split, frame in split_frames.items()}
    test_track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()

    models = [str(value) for value in settings["models"]]
    ensemble = settings["ensemble"]
    ensemble_name = str(ensemble["name"])
    components = [str(value) for value in ensemble["components"]]
    weights = np.asarray(ensemble["weights"], dtype=np.float64)
    if set(components) - set(models) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid locked ensemble configuration")

    metrics_rows, history_frames = [], []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {model: {c: [] for c in CONDITIONS} for model in models + [ensemble_name]}
    validation_predictions: dict[str, list[np.ndarray]] = {model: [] for model in models}
    threshold_values: dict[str, list[np.ndarray]] = {model: [] for model in models + [ensemble_name]}
    for seed in [int(value) for value in settings["seeds"]]:
        for model_name in models:
            model, thresholds, history = train_model(model_name, seed, features, availability, reliability, target, settings, device)
            history_frames.append(history)
            threshold_values[model_name].append(thresholds)
            validation_predictions[model_name].append(
                predict(model, features["validation"], availability["validation"], reliability["validation"], device, int(settings["evaluation_batch_size"]))
            )
            for number, condition in enumerate(CONDITIONS):
                condition_availability = intervention(condition, availability["test"], 20260910 + number)
                probability = predict(model, features["test"], condition_availability, reliability["test"], device, int(settings["evaluation_batch_size"]))
                predictions[model_name][condition].append(probability)
                metrics_rows.append(
                    {"model": model_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, thresholds)}
                )
            print(f"complete model={model_name} seed={seed}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Locked-recipe ensemble for this seed: thresholds from validation only.
        seed_index = len(threshold_values[ensemble_name])
        validation_probability = sum(
            float(weight) * validation_predictions[component][seed_index] for component, weight in zip(components, weights, strict=True)
        )
        ensemble_thresholds = tune_thresholds(target["validation"], validation_probability)
        threshold_values[ensemble_name].append(ensemble_thresholds)
        for condition in CONDITIONS:
            probability = sum(
                float(weight) * predictions[component][condition][seed_index] for component, weight in zip(components, weights, strict=True)
            )
            predictions[ensemble_name][condition].append(probability)
            metrics_rows.append(
                {"model": ensemble_name, "seed": seed, "condition": condition, **metric_values(target["test"], probability, ensemble_thresholds)}
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)

    mean_probabilities = {
        model: {condition: np.mean(values, axis=0) for condition, values in condition_values.items()}
        for model, condition_values in predictions.items()
    }
    mean_thresholds = {model: np.mean(values, axis=0) for model, values in threshold_values.items()}
    per_label, tests = paired_tests(
        target["test"], labels, mean_probabilities, mean_thresholds, proposed=ensemble_name, replicates=int(settings["bootstrap_replicates"])
    )
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)
    tests.to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    prediction_frames = []
    for model, condition_values in mean_probabilities.items():
        for condition, probability in condition_values.items():
            frame = pd.DataFrame(probability, columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", test_track_ids)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", model)
            prediction_frames.append(frame)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)

    coverage = {
        split: {modality: float(availability[split][:, index].mean()) for index, modality in enumerate(MODALITIES)}
        for split in split_frames
    }
    (output_root / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "environment": environment_record(device),
        "task": "E3 FMA artist-disjoint multi-label genre tagging (locked Onion recipe)",
        "samples": {split: len(frame) for split, frame in split_frames.items()},
        "artists": {split: int(frame["artist_id"].nunique()) for split, frame in split_frames.items()},
        "labels": len(labels),
        "modalities": list(MODALITIES),
        "coverage": coverage,
        "conditions": CONDITIONS,
        "recipe": "architectures, dropout rates and 0.6/0.4 ensemble weights locked from Onion; only thresholds calibrated on FMA validation",
        "statistical_unit": "label; mean prediction across seeds; paired bootstrap with Holm correction",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completion = {"status": "complete", "elapsed_seconds": time.perf_counter() - started, "run_directory": str(output_root)}
    (output_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.overwrite)


if __name__ == "__main__":
    main()
