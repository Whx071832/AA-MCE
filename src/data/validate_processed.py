"""Validate core invariants of the prepared SocialMusicFusion data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root / "Data" / "processed" / "v1"
    failures: list[str] = []
    checks: dict[str, object] = {}
    for relative in [
        "entities/aa_artists.parquet", "entities/aa_albums.parquet", "entities/onion_tracks.parquet",
        "social/musicsem_social_semantics.parquet", "external/fma_tracks.parquet",
    ]:
        path = root / relative
        if not path.exists():
            failures.append(f"missing {relative}")
    if not failures:
        onion_tracks = pd.read_parquet(root / "entities/onion_tracks.parquet")
        checks["onion_track_count"] = len(onion_tracks)
        checks["onion_track_id_unique"] = bool(onion_tracks["track_id"].is_unique)
        if not onion_tracks["track_id"].is_unique:
            failures.append("onion track ids are not unique")
        social = pd.read_parquet(root / "social/musicsem_social_semantics.parquet")
        checks["musicsem_rows"] = len(social)
        for thread, group in social.groupby("thread_id", dropna=False):
            if group["thread_split"].nunique(dropna=False) > 1:
                failures.append(f"thread split leakage for {thread!r}")
                break
        for modality in ["audio_essentia", "lyrics_tfidf", "visual_resnet"]:
            path = root / "features/onion" / modality / "index.parquet"
            if path.exists():
                index = pd.read_parquet(path)
                checks[f"{modality}_rows"] = len(index)
                if not index["track_id"].is_unique:
                    failures.append(f"duplicate ids in {modality} index")
                part = root / "features/onion" / modality / "part-00000.npy"
                if part.exists():
                    matrix = np.load(part, allow_pickle=False, mmap_mode="r")
                    checks[f"{modality}_dimension"] = int(matrix.shape[1])
        alignment_path = root / "alignments/musicsem_to_onion.parquet"
        if alignment_path.exists():
            alignment = pd.read_parquet(alignment_path, columns=["source_row_id", "matched_track_id", "method"])
            checks["musicsem_alignment_rows"] = len(alignment)
            checks["musicsem_alignment_source_rows_unique"] = bool(alignment["source_row_id"].is_unique)
            if not alignment["source_row_id"].is_unique:
                failures.append("MusicSem alignment source rows are not unique")
            known_tracks = set(onion_tracks["track_id"])
            linked = set(alignment.loc[alignment["matched_track_id"].notna(), "matched_track_id"])
            checks["musicsem_alignment_linked_rows"] = int(alignment["matched_track_id"].notna().sum())
            if not linked.issubset(known_tracks):
                failures.append("MusicSem alignment contains an unknown Onion track id")
        artist_split_path = root / "entities/aa_artist_cold_start_splits.parquet"
        track_split_path = root / "entities/onion_track_artist_disjoint_splits.parquet"
        if artist_split_path.exists() and track_split_path.exists():
            artist_splits = pd.read_parquet(artist_split_path)
            track_splits = pd.read_parquet(track_split_path)
            checks["artist_cold_split_counts"] = artist_splits["cold_start_split"].value_counts().to_dict()
            if artist_splits["artist_id"].duplicated().any():
                failures.append("artist cold-start split has duplicate artist ids")
            mapped_tracks = track_splits.dropna(subset=["artist_id", "cold_start_split"])
            expected = mapped_tracks.merge(artist_splits, on="artist_id", suffixes=("_track", "_artist"))
            if not (expected["cold_start_split_track"] == expected["cold_start_split_artist"]).all():
                failures.append("artist-disjoint track split disagrees with artist split")
    temporal_manifest = root / "interactions/temporal_split_manifest.json"
    if temporal_manifest.exists():
        manifest = json.loads(temporal_manifest.read_text(encoding="utf-8"))
        checks["interaction_split_rows"] = manifest["split_rows"]
        split_root = root / "interactions/temporal_split"
        observed: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            files = sorted((split_root / f"split={split}").glob("user_bucket=*/events.parquet"))
            observed[split] = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
        checks["interaction_split_rows_observed"] = observed
        if observed != manifest["split_rows"]:
            failures.append("interaction split parquet row totals do not match manifest")
        # Deterministic bucket sample: enough to catch a broken local sort while
        # retaining a fast validator for this 253M-event dataset.
        sampled_buckets = (0, 13, 31)
        ordering_ok = True
        for bucket in sampled_buckets:
            parts = []
            for split in ("train", "validation", "test"):
                file = split_root / f"split={split}" / f"user_bucket={bucket}" / "events.parquet"
                if file.exists():
                    frame = pd.read_parquet(file, columns=["user_id", "timestamp"])
                    frame["split"] = split
                    parts.append(frame)
            if not parts:
                ordering_ok = False
                break
            bucket_frame = pd.concat(parts, ignore_index=True)
            extrema = bucket_frame.groupby(["user_id", "split"])["timestamp"].agg(["min", "max"]).unstack("split")
            train_to_validation = (extrema[("max", "train")] <= extrema[("min", "validation")]).fillna(True)
            validation_to_test = (extrema[("max", "validation")] <= extrema[("min", "test")]).fillna(True)
            if not bool((train_to_validation & validation_to_test).all()):
                ordering_ok = False
                break
        checks["interaction_temporal_order_sample_buckets"] = list(sampled_buckets)
        checks["interaction_temporal_order_valid"] = ordering_ok
        if not ordering_ok:
            failures.append("temporal ordering check failed in sampled interaction bucket")
    print(json.dumps({"valid": not failures, "checks": checks, "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
