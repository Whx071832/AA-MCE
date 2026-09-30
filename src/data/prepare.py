"""Build auditable, experiment-ready tables for SocialMusicFusion.

The input datasets remain immutable under ``Data/``.  This module writes only
to ``Data/processed/v1`` and is safe to re-run stage by stage.  It deliberately
never deserializes the untrusted MusicSem pickle archive; only ``train.csv`` is
read.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import csv
import hashlib
import html
import json
import re
import shutil
import tarfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse


# ``datetime.UTC`` exists only on Python >= 3.11; it is an alias of ``timezone.utc``.
UTC = timezone.utc
VERSION = "v1"
SEED = 20260724
SEMANTIC_COLUMNS = ("descriptive", "contextual", "situational", "atmospheric", "metadata")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
REDDIT_USER_RE = re.compile(r"(?<!\w)(?:/u/|u/)[A-Za-z0-9_-]+", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
SPOTIFY_TRACK_RE = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"\s*(?:\(|\[)\s*(?:live|remaster(?:ed)?|radio edit|edit|version|mono|stereo).*?(?:\)|\])\s*$", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_text(value: Any, *, redact_pii: bool = False) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = html.unescape(str(value))
    text = HTML_RE.sub(" ", text)
    if redact_pii:
        text = URL_RE.sub("<URL>", text)
        text = EMAIL_RE.sub("<EMAIL>", text)
        text = REDDIT_USER_RE.sub("<REDDIT_USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def safe_list(value: Any) -> list[str]:
    """Normalize JSON/Python-list-like metadata without evaluating arbitrary code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        result = value
    elif isinstance(value, tuple):
        result = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                result = [text]
    if not isinstance(result, (list, tuple)):
        result = [result]
    return [str(item).strip() for item in result if item is not None and str(item).strip()]


def safe_mapping(value: Any) -> dict[str, Any]:
    """Return a mapping only when the source field actually is a mapping."""
    return value if isinstance(value, dict) else {}


def stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def source_file_record(path: Path, root: Path, *, with_sha256: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0).isoformat(),
    }
    if with_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    return record


class Pipeline:
    def __init__(self, project_root: Path, *, overwrite: bool, checksum: bool) -> None:
        self.project_root = project_root.resolve()
        self.data = self.project_root / "Data"
        self.output = self.data / "processed" / VERSION
        self.overwrite = overwrite
        self.checksum = checksum
        self.output.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, Any] = {"schema_version": VERSION, "created_at": now_iso(), "stages": {}}

    def output_path(self, *parts: str) -> Path:
        path = self.output.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def require_new(self, path: Path) -> bool:
        if not path.exists():
            return True
        if self.overwrite:
            if path.is_dir():
                resolved = path.resolve()
                allowed = self.output.resolve()
                if allowed not in resolved.parents:
                    raise RuntimeError(f"Refusing to delete outside processed output: {resolved}")
                shutil.rmtree(resolved)
            else:
                path.unlink()
            return True
        print(f"[skip] exists: {path.relative_to(self.project_root)}")
        return False

    def extract_lyrics(self) -> None:
        archive = self.data / "music4all_onion" / "processed_lyrics.tar.gz"
        destination = self.data / "music4all_onion" / "processed_lyrics"
        if destination.exists() and any(destination.glob("*.txt")):
            self.stats["stages"]["extract_lyrics"] = {"status": "already_extracted", "destination": str(destination.relative_to(self.data))}
            return
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            base = (self.data / "music4all_onion").resolve()
            for member in members:
                candidate = (base / member.name).resolve()
                if base not in candidate.parents and candidate != base:
                    raise RuntimeError(f"Unsafe path in archive: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Links are not permitted in archive: {member.name}")
            tar.extractall(base, members=members, filter="data")
        count = sum(1 for _ in destination.glob("*.txt"))
        self.stats["stages"]["extract_lyrics"] = {"status": "extracted", "files": count, "destination": str(destination.relative_to(self.data))}

    def prepare_music4all_aa(self) -> None:
        entity_dir = self.output / "entities"
        target = entity_dir / "aa_artists.parquet"
        if not self.require_new(target):
            return
        aa_root = self.data / "music4all_aa"
        artist_splits = read_json(aa_root / "artist_modality_splits.json")
        album_splits = read_json(aa_root / "album_modality_splits.json")

        def modality_protocol(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[int]], list[dict[str, Any]]]:
            train_ids = {str(entity_id) for entity_id in safe_list(payload.get("train"))}
            availability: dict[str, list[int]] = {}
            long_rows: list[dict[str, Any]] = []
            nested = safe_mapping(payload.get("missing_modality"))
            for percentage_text, entity_ids in nested.items():
                try:
                    percentage = int(percentage_text)
                except (TypeError, ValueError):
                    continue
                for entity_id in safe_list(entity_ids):
                    entity_id = str(entity_id)
                    availability.setdefault(entity_id, []).append(percentage)
                    long_rows.append({"entity_id": entity_id, "modality_availability_percent": percentage})
            primary = {entity_id: "train" for entity_id in train_ids}
            primary.update({entity_id: "missing_modality" for entity_id in availability})
            return primary, {entity_id: sorted(set(values), reverse=True) for entity_id, values in availability.items()}, long_rows

        artist_split, artist_conditions, artist_condition_rows = modality_protocol(artist_splits)
        album_split, album_conditions, album_condition_rows = modality_protocol(album_splits)

        artists: list[dict[str, Any]] = []
        artist_tracks: list[dict[str, str]] = []
        artist_errors: list[dict[str, str]] = []
        for file in sorted((aa_root / "artists_json").glob("*.json")):
            try:
                payload = read_json(file)
                artist = payload["artist_info"]["artist"]
                artist_id = str(payload.get("mbid") or artist.get("id") or file.stem)
                image_url = payload.get("artist_image_url")
                wiki = clean_text(safe_mapping(artist.get("wiki")).get("summary"))
                artists.append(
                    {
                        "artist_id": artist_id,
                        "name": artist.get("name"),
                        "name_normalized": normalize_name(artist.get("name")),
                        "sort_name": artist.get("sort-name"),
                        "country": artist.get("country"),
                        "artist_type": artist.get("type"),
                        "life_start": safe_mapping(artist.get("life-span")).get("begin"),
                        "life_end": safe_mapping(artist.get("life-span")).get("end"),
                        "genres": safe_list(artist.get("genres")),
                        "text": wiki,
                        "image_url": image_url,
                        "has_text": bool(wiki),
                        "has_image": bool(image_url),
                        "modality_split": artist_split.get(artist_id, "unassigned"),
                        "modality_availability_percentages": artist_conditions.get(artist_id, []),
                        "source_file": file.name,
                    }
                )
                for track_id in payload.get("music4all_onion_id", []):
                    artist_tracks.append({"track_id": str(track_id), "artist_id": artist_id})
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                artist_errors.append({"file": file.name, "error": repr(error)})

        albums: list[dict[str, Any]] = []
        album_tracks: list[dict[str, Any]] = []
        album_errors: list[dict[str, str]] = []
        for file in sorted((aa_root / "album_json").glob("*.json")):
            try:
                payload = read_json(file)
                album = payload["album_info"]["album"]
                album_id = str(payload.get("mbid") or album.get("mbid") or file.stem)
                tracks_container = safe_mapping(album.get("tracks"))
                raw_track_items = tracks_container.get("track", [])
                if isinstance(raw_track_items, dict):
                    raw_track_items = [raw_track_items]
                if not isinstance(raw_track_items, list):
                    raw_track_items = []
                first_artist_id = None
                if isinstance(raw_track_items, list) and raw_track_items:
                    first_artist_id = safe_mapping(raw_track_items[0]).get("artist", {})
                    first_artist_id = safe_mapping(first_artist_id).get("mbid")
                tags = safe_mapping(album.get("tags")).get("tag", [])
                if isinstance(tags, dict):
                    tags = [tags]
                tag_names = [str(tag.get("name")) for tag in tags if isinstance(tag, dict) and tag.get("name")]
                wiki = safe_mapping(album.get("wiki"))
                summary = clean_text(wiki.get("content") or wiki.get("summary"))
                image_urls = [str(value) for value in payload.get("album_image_url", []) if value]
                albums.append(
                    {
                        "album_id": album_id,
                        "artist_id": first_artist_id,
                        "artist_name": album.get("artist"),
                        "artist_name_normalized": normalize_name(album.get("artist")),
                        "title": album.get("name"),
                        "title_normalized": normalize_name(album.get("name")),
                        "release_date": payload.get("release_date"),
                        "genres": safe_list(album.get("genres")),
                        "tags": tag_names,
                        "text": summary,
                        "image_urls": image_urls,
                        "has_text": bool(summary),
                        "has_image": bool(image_urls),
                        "modality_split": album_split.get(album_id, "unassigned"),
                        "modality_availability_percentages": album_conditions.get(album_id, []),
                        "source_file": file.name,
                    }
                )
                onion_ids = [str(item) for item in payload.get("music4all_onion_id", [])]
                for rank, track_id in enumerate(onion_ids, start=1):
                    source_track = safe_mapping(raw_track_items[rank - 1]) if rank <= len(raw_track_items) else {}
                    album_tracks.append(
                        {
                            "track_id": track_id,
                            "album_id": album_id,
                            "artist_id": safe_mapping(source_track.get("artist")).get("mbid") or first_artist_id,
                            "track_title": source_track.get("name"),
                            "track_title_normalized": normalize_name(source_track.get("name")),
                            "track_rank": rank,
                            "duration_seconds": source_track.get("duration"),
                        }
                    )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                album_errors.append({"file": file.name, "error": repr(error)})

        entity_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(artists).to_parquet(target, index=False)
        pd.DataFrame(albums).to_parquet(entity_dir / "aa_albums.parquet", index=False)
        pd.DataFrame(artist_tracks).drop_duplicates().to_parquet(entity_dir / "aa_track_artists.parquet", index=False)
        pd.DataFrame(album_tracks).drop_duplicates().to_parquet(entity_dir / "aa_track_albums.parquet", index=False)
        pd.DataFrame(artist_condition_rows).rename(columns={"entity_id": "artist_id"}).to_parquet(entity_dir / "aa_artist_modality_conditions.parquet", index=False)
        pd.DataFrame(album_condition_rows).rename(columns={"entity_id": "album_id"}).to_parquet(entity_dir / "aa_album_modality_conditions.parquet", index=False)
        write_json(entity_dir / "aa_parse_errors.json", {"artists": artist_errors, "albums": album_errors})
        self.stats["stages"]["music4all_aa"] = {
            "artists": len(artists), "albums": len(albums), "artist_track_edges": len(artist_tracks),
            "album_track_edges": len(album_tracks), "artist_parse_errors": len(artist_errors),
            "album_parse_errors": len(album_errors),
        }

    def prepare_musicsem(self) -> None:
        target = self.output / "social" / "musicsem_social_semantics.parquet"
        if not self.require_new(target):
            return
        source = self.data / "musicsem" / "train.csv"
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:") or column == ""]
        frame = frame.drop(columns=unnamed, errors="ignore")
        records: list[dict[str, Any]] = []
        thread_splits: dict[str, str] = {}
        for source_row_id, row in enumerate(frame.to_dict(orient="records")):
            # In this release ``thread`` contains only the subreddit/community
            # (five values), not a post-level identifier.  Exact source-post
            # text is therefore used exclusively to derive a deterministic,
            # non-reversible thread id for group splitting.
            community = str(row.get("thread") or "").strip() or None
            source_post = clean_text(row.get("raw_text")) or str(row.get("pairs") or "")
            thread_id = "ms_" + hashlib.sha256(source_post.encode("utf-8")).hexdigest()[:24]
            if thread_id not in thread_splits:
                bucket = stable_bucket(thread_id, 10)
                thread_splits[thread_id] = "train" if bucket < 7 else "validation" if bucket < 8 else "test"
            spotify_link = row.get("spotify_link") or None
            match = SPOTIFY_TRACK_RE.search(spotify_link or "")
            semantic_values = {column: safe_list(row.get(column)) for column in SEMANTIC_COLUMNS}
            raw_text = clean_text(row.get("raw_text"), redact_pii=True)
            prompt = clean_text(row.get("prompt"), redact_pii=True)
            records.append(
                {
                    "source_row_id": source_row_id,
                    "social_id": row.get("unique_id") or None,
                    "thread_id": thread_id,
                    "community": community,
                    "thread_split": thread_splits[thread_id],
                    "spotify_track_id": match.group(1) if match else None,
                    "song": row.get("song") or None,
                    "song_normalized": normalize_name(row.get("song")),
                    "artist": row.get("artist") or None,
                    "artist_normalized": normalize_name(row.get("artist")),
                    "raw_text": raw_text,
                    "prompt": prompt,
                    **semantic_values,
                    "has_semantic_label": any(semantic_values.values()),
                }
            )
        social_dir = target.parent
        social_dir.mkdir(parents=True, exist_ok=True)
        output = pd.DataFrame(records)
        output.to_parquet(target, index=False)
        output.loc[:, ["source_row_id", "social_id", "thread_id", "thread_split", "song", "artist", "prompt"]].to_parquet(
            social_dir / "musicsem_retrieval_queries.parquet", index=False
        )
        write_json(social_dir / "musicsem_thread_splits.json", {"seed": SEED, "thread_id_rule": "sha256(raw_text)[:24]", "thread_count": len(thread_splits), "splits": Counter(thread_splits.values())})
        self.stats["stages"]["musicsem"] = {
            "rows": len(output), "unique_threads": len(thread_splits),
            "unique_songs": int(output["song_normalized"].nunique(dropna=True)),
            "unique_artists": int(output["artist_normalized"].nunique(dropna=True)),
            "semantic_label_rate": float(output["has_semantic_label"].mean()),
            "split_counts": dict(Counter(thread_splits.values())),
        }

    def prepare_fma(self) -> None:
        target = self.output / "external" / "fma_tracks.parquet"
        if not self.require_new(target):
            return
        metadata = self.data / "fma" / "fma_metadata" / "fma_metadata"
        audio_root = self.data / "fma" / "fma_small" / "fma_small"
        tracks = pd.read_csv(metadata / "tracks.csv", header=[0, 1], index_col=0, low_memory=False)
        tracks.columns = [f"{first}_{second}".rstrip("_") for first, second in tracks.columns]
        tracks.index.name = "track_id"
        tracks = tracks.reset_index()
        def column(name: str, default: Any = None) -> pd.Series:
            return tracks[name] if name in tracks.columns else pd.Series(default, index=tracks.index)
        track_ids = tracks["track_id"].astype(int)
        relative_paths = track_ids.map(lambda track_id: f"fma_small/fma_small/{track_id // 1000:03d}/{track_id:06d}.mp3")
        audio_available = relative_paths.map(lambda rel: (self.data / "fma" / rel).is_file())
        output = pd.DataFrame(
            {
                "track_id": track_ids.astype(str),
                "source_track_id": track_ids,
                "title": column("track_title"),
                "title_normalized": column("track_title").map(normalize_name),
                "artist_id": column("artist_id").astype("Int64").astype(str).replace("<NA>", None),
                "artist_name": column("artist_name"),
                "artist_name_normalized": column("artist_name").map(normalize_name),
                "album_id": column("album_id").astype("Int64").astype(str).replace("<NA>", None),
                "album_title": column("album_title"),
                "genres": column("track_genres").map(safe_list),
                "genres_all": column("track_genres_all").map(safe_list),
                "genre_top": column("track_genre_top"),
                "tags": column("track_tags").map(safe_list),
                "duration_seconds": pd.to_numeric(column("track_duration"), errors="coerce").astype("Int32"),
                "language_code": column("track_language_code"),
                "license": column("track_license"),
                "subset": column("set_subset"),
                "official_split": column("set_split"),
                "audio_path": relative_paths.where(audio_available, None),
                "audio_available": audio_available,
            }
        )
        external = target.parent
        external.mkdir(parents=True, exist_ok=True)
        output.to_parquet(target, index=False)
        output[["artist_id", "artist_name", "artist_name_normalized"]].drop_duplicates().dropna(subset=["artist_id"]).to_parquet(external / "fma_artists.parquet", index=False)
        output[["album_id", "album_title", "artist_id"]].drop_duplicates().dropna(subset=["album_id"]).to_parquet(external / "fma_albums.parquet", index=False)
        genres = pd.read_csv(metadata / "genres.csv", index_col=0)
        genres.reset_index().to_parquet(external / "fma_genres.parquet", index=False)
        self.stats["stages"]["fma"] = {
            "tracks": len(output), "audio_available": int(audio_available.sum()),
            "audio_missing": int((~audio_available).sum()), "artists": int(output["artist_id"].nunique(dropna=True)),
        }

    def _write_dense_feature(self, source: Path, name: str, *, rows_per_shard: int) -> dict[str, Any]:
        feature_root = self.output / "features" / "onion" / name
        index_path = feature_root / "index.parquet"
        if not self.require_new(feature_root):
            return {"status": "skipped"}
        feature_root.mkdir(parents=True, exist_ok=True)
        with bz2.open(source, "rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            feature_names = header[1:]
            width = len(feature_names)
            ids: list[str] = []
            values: list[np.ndarray] = []
            index_frames: list[pd.DataFrame] = []
            malformed = 0
            shard = 0
            row_total = 0

            def flush() -> None:
                nonlocal shard, row_total, ids, values
                if not ids:
                    return
                matrix = np.vstack(values).astype(np.float32, copy=False)
                shard_name = f"part-{shard:05d}"
                np.save(feature_root / f"{shard_name}.npy", matrix, allow_pickle=False)
                np.save(feature_root / f"{shard_name}.track_ids.npy", np.asarray(ids, dtype="U32"), allow_pickle=False)
                index_frames.append(pd.DataFrame({"track_id": ids, "shard": shard, "row_offset": np.arange(len(ids), dtype=np.int32)}))
                row_total += len(ids)
                shard += 1
                ids, values = [], []

            for line in handle:
                track_id, separator, value_text = line.rstrip("\r\n").partition("\t")
                if not separator or not track_id:
                    malformed += 1
                    continue
                array = np.fromstring(value_text, sep="\t", dtype=np.float32)
                if len(array) != width:
                    malformed += 1
                    continue
                ids.append(track_id)
                values.append(array)
                if len(ids) >= rows_per_shard:
                    flush()
            flush()
        pd.concat(index_frames, ignore_index=True).to_parquet(index_path, index=False)
        write_json(feature_root / "metadata.json", {"source": source.name, "rows": row_total, "dimension": width, "dtype": "float32", "feature_names": feature_names, "malformed_rows": malformed, "rows_per_shard": rows_per_shard})
        return {"rows": row_total, "dimension": width, "malformed_rows": malformed, "shards": shard}

    def prepare_onion_features(self) -> None:
        onion = self.data / "music4all_onion"
        specifications = (
            ("audio_essentia", "id_essentia.tsv.bz2", 8192),
            ("lyrics_tfidf", "id_lyrics_tf-idf.tsv.bz2", 8192),
            ("visual_resnet", "id_resnet.tsv.bz2", 2048),
            # Second view per sense for the six-view extended-modality study
            # (E7): existing feature roots are skipped unless --overwrite.
            ("audio_ivec1024", "id_ivec1024.tsv.bz2", 8192),
            ("lyrics_word2vec", "id_lyrics_word2vec.tsv.bz2", 8192),
            ("visual_incp", "id_incp.tsv.bz2", 2048),
        )
        results: dict[str, Any] = {}
        for name, source_name, rows_per_shard in specifications:
            print(f"[features] start {name}", flush=True)
            results[name] = self._write_dense_feature(onion / source_name, name, rows_per_shard=rows_per_shard)
            print(f"[features] complete {name}: {results[name]}", flush=True)
        self.stats["stages"]["onion_dense_features"] = results

    def prepare_onion_tags_and_tracks(self) -> None:
        feature_root = self.output / "features" / "onion" / "tag_weights"
        track_target = self.output / "entities" / "onion_tracks.parquet"
        if not self.require_new(feature_root):
            return
        feature_root.mkdir(parents=True, exist_ok=True)
        source = self.data / "music4all_onion" / "id_tags_dict.tsv.bz2"
        tag_to_column: dict[str, int] = {}
        track_to_row: dict[str, int] = {}
        track_ids: list[str] = []
        tag_weights_by_track: list[dict[str, float]] = []
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        malformed = 0
        with bz2.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            for row_index, row in enumerate(reader):
                if len(row) < 2:
                    malformed += 1
                    continue
                track_id = row[0].strip()
                try:
                    parsed = ast.literal_eval(row[1])
                except (ValueError, SyntaxError):
                    malformed += 1
                    continue
                if not isinstance(parsed, dict):
                    malformed += 1
                    continue
                if track_id not in track_to_row:
                    track_to_row[track_id] = len(track_ids)
                    track_ids.append(track_id)
                    tag_weights_by_track.append({})
                track_row = track_to_row[track_id]
                for tag, weight in parsed.items():
                    tag_name = clean_text(tag)
                    if not tag_name:
                        continue
                    column = tag_to_column.setdefault(tag_name, len(tag_to_column))
                    try:
                        score = float(weight)
                    except (TypeError, ValueError):
                        score = 1.0
                    previous = tag_weights_by_track[track_row].get(tag_name)
                    if previous is None or score > previous:
                        tag_weights_by_track[track_row][tag_name] = score
        for track_row, tag_weights in enumerate(tag_weights_by_track):
            for tag_name, score in tag_weights.items():
                rows.append(track_row)
                columns.append(tag_to_column[tag_name])
                weights.append(score)
        tag_lists = [list(tag_weights) for tag_weights in tag_weights_by_track]
        matrix = sparse.csr_matrix((np.asarray(weights, dtype=np.float32), (rows, columns)), shape=(len(track_ids), len(tag_to_column)), dtype=np.float32)
        sparse.save_npz(feature_root / "matrix.npz", matrix, compressed=True)
        np.save(feature_root / "track_ids.npy", np.asarray(track_ids, dtype="U32"), allow_pickle=False)
        vocabulary = [None] * len(tag_to_column)
        for tag, index in tag_to_column.items():
            vocabulary[index] = tag
        write_json(feature_root / "vocabulary.json", vocabulary)
        pd.DataFrame({"track_id": track_ids, "tags": tag_lists, "tag_count": [len(tags) for tags in tag_lists]}).to_parquet(feature_root / "track_tags.parquet", index=False)

        audio_ids = pd.read_parquet(self.output / "features" / "onion" / "audio_essentia" / "index.parquet", columns=["track_id"])
        lyric_ids = pd.read_parquet(self.output / "features" / "onion" / "lyrics_tfidf" / "index.parquet", columns=["track_id"])
        visual_ids = pd.read_parquet(self.output / "features" / "onion" / "visual_resnet" / "index.parquet", columns=["track_id"])
        all_ids = pd.DataFrame({"track_id": sorted(set(audio_ids["track_id"]) | set(lyric_ids["track_id"]) | set(visual_ids["track_id"]) | set(track_ids))})
        tracks = all_ids.merge(pd.DataFrame({"track_id": track_ids, "tags": tag_lists}), on="track_id", how="left")
        tracks["has_audio"] = tracks["track_id"].isin(set(audio_ids["track_id"]))
        tracks["has_lyrics_feature"] = tracks["track_id"].isin(set(lyric_ids["track_id"]))
        tracks["has_visual_feature"] = tracks["track_id"].isin(set(visual_ids["track_id"]))
        albums_path = self.output / "entities" / "aa_track_albums.parquet"
        artists_path = self.output / "entities" / "aa_track_artists.parquet"
        if albums_path.exists():
            album_edges = pd.read_parquet(albums_path).sort_values("track_rank").drop_duplicates("track_id")
            tracks = tracks.merge(album_edges[["track_id", "album_id", "artist_id", "track_title", "track_title_normalized"]], on="track_id", how="left")
        if artists_path.exists():
            artist_edges = pd.read_parquet(artists_path).drop_duplicates("track_id")
            tracks = tracks.merge(artist_edges.rename(columns={"artist_id": "artist_id_from_artist"}), on="track_id", how="left")
            if "artist_id" in tracks.columns:
                tracks["artist_id"] = tracks["artist_id"].fillna(tracks["artist_id_from_artist"])
                tracks = tracks.drop(columns=["artist_id_from_artist"])
            else:
                tracks = tracks.rename(columns={"artist_id_from_artist": "artist_id"})
        tracks.to_parquet(track_target, index=False)
        self.stats["stages"]["onion_tags_tracks"] = {
            "tracks": len(tracks), "tag_vocabulary": len(vocabulary), "nonzero_tag_edges": int(matrix.nnz),
            "malformed_rows": malformed, "audio_available": int(tracks["has_audio"].sum()),
            "lyrics_available": int(tracks["has_lyrics_feature"].sum()), "visual_available": int(tracks["has_visual_feature"].sum()),
        }

    def prepare_entity_alignment_and_splits(self) -> None:
        """Create conservative MusicSem--Onion links and artist-disjoint splits.

        A+A is the only supplied source that connects human-readable titles and
        artists to Onion ids.  Exact matches are accepted only when unique;
        fuzzy matches require a high score and a score margin.  All borderline
        records are retained for review instead of being force-linked.
        """
        alignment_dir = self.output / "alignments"
        target = alignment_dir / "musicsem_to_onion.parquet"
        if not self.require_new(target):
            return
        entities = self.output / "entities"
        social = pd.read_parquet(self.output / "social" / "musicsem_social_semantics.parquet")
        aa_artists = pd.read_parquet(entities / "aa_artists.parquet", columns=["artist_id", "name", "name_normalized"])
        aa_albums = pd.read_parquet(entities / "aa_albums.parquet", columns=["album_id", "artist_id", "artist_name", "artist_name_normalized"])
        aa_tracks = pd.read_parquet(entities / "aa_track_albums.parquet")
        candidates = aa_tracks.merge(aa_albums, on="album_id", how="left", suffixes=("", "_album"))
        candidates = candidates.merge(aa_artists, on="artist_id", how="left", suffixes=("", "_artist"))
        candidates["candidate_artist"] = candidates["name"].fillna(candidates["artist_name"])
        candidates["candidate_artist_normalized"] = candidates["name_normalized"].fillna(candidates["artist_name_normalized"])
        candidates = candidates.loc[
            candidates["track_id"].notna()
            & candidates["track_title_normalized"].notna()
            & candidates["candidate_artist_normalized"].notna(),
            ["track_id", "track_title", "track_title_normalized", "candidate_artist", "candidate_artist_normalized", "album_id", "artist_id"],
        ].drop_duplicates()
        candidates = candidates.sort_values(["track_id", "album_id"], kind="stable").drop_duplicates("track_id")
        exact_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        artist_index: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates.to_dict(orient="records"):
            exact_index.setdefault((candidate["candidate_artist_normalized"], candidate["track_title_normalized"]), []).append(candidate)
            artist_index.setdefault(candidate["candidate_artist_normalized"], []).append(candidate)

        def similarity(left: str, right: str) -> float:
            sequence = SequenceMatcher(None, left, right).ratio()
            left_tokens, right_tokens = set(left.split()), set(right.split())
            jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens or right_tokens else 0.0
            return 0.7 * sequence + 0.3 * jaccard

        matches: list[dict[str, Any]] = []
        for row in social.to_dict(orient="records"):
            artist = row.get("artist_normalized")
            title = row.get("song_normalized")
            base: dict[str, Any] = {
                "source_row_id": row["source_row_id"], "social_id": row.get("social_id"), "song": row.get("song"),
                "artist": row.get("artist"), "song_normalized": title, "artist_normalized": artist,
                "matched_track_id": None, "matched_album_id": None, "matched_artist_id": None,
                "method": "unmatched", "confidence": 0.0, "candidate_count": 0,
            }
            if not artist or not title:
                base["method"] = "missing_name"
                matches.append(base)
                continue
            exact = exact_index.get((artist, title), [])
            if len(exact) == 1:
                candidate = exact[0]
                base.update({"matched_track_id": candidate["track_id"], "matched_album_id": candidate["album_id"], "matched_artist_id": candidate["artist_id"], "method": "exact", "confidence": 1.0, "candidate_count": 1})
                matches.append(base)
                continue
            if len(exact) > 1:
                base.update({"method": "ambiguous_exact", "confidence": 1.0, "candidate_count": len(exact)})
                matches.append(base)
                continue
            scored = sorted(
                ((similarity(title, candidate["track_title_normalized"]), candidate) for candidate in artist_index.get(artist, [])),
                key=lambda item: (-item[0], item[1]["track_id"]),
            )
            if not scored:
                matches.append(base)
                continue
            score, candidate = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            base.update({"candidate_count": len(scored), "confidence": float(score)})
            if score >= 0.92 and score - runner_up >= 0.03:
                base.update({"matched_track_id": candidate["track_id"], "matched_album_id": candidate["album_id"], "matched_artist_id": candidate["artist_id"], "method": "fuzzy_high_confidence"})
            elif score >= 0.80:
                base["method"] = "manual_review"
            matches.append(base)
        alignments = pd.DataFrame(matches)
        alignment_dir.mkdir(parents=True, exist_ok=True)
        alignments.to_parquet(target, index=False)
        review = alignments.loc[alignments["method"] == "manual_review"].copy()
        if not review.empty:
            review["sample_key"] = review["source_row_id"].map(lambda value: stable_bucket(str(value), 1_000_000))
            review = review.sort_values(["sample_key", "source_row_id"]).head(500).drop(columns="sample_key")
        review.to_csv(alignment_dir / "musicsem_alignment_review_sample.csv", index=False, encoding="utf-8")

        artist_splits = aa_artists[["artist_id"]].copy()
        artist_splits["cold_start_split"] = artist_splits["artist_id"].map(
            lambda value: "train" if stable_bucket(str(value), 10) < 8 else "validation" if stable_bucket(str(value), 10) < 9 else "test"
        )
        artist_splits.to_parquet(entities / "aa_artist_cold_start_splits.parquet", index=False)
        track_splits = pd.read_parquet(entities / "onion_tracks.parquet").merge(artist_splits, on="artist_id", how="left")
        track_splits[["track_id", "artist_id", "album_id", "cold_start_split"]].to_parquet(entities / "onion_track_artist_disjoint_splits.parquet", index=False)
        write_json(alignment_dir / "alignment_manifest.json", {
            "candidate_tracks": len(candidates), "threshold": 0.92, "minimum_margin": 0.03,
            "match_counts": dict(alignments["method"].value_counts()), "review_sample_rows": len(review),
            "artist_split_counts": dict(artist_splits["cold_start_split"].value_counts()),
        })
        self.stats["stages"]["entity_alignment"] = {
            "candidate_tracks": len(candidates), "match_counts": dict(alignments["method"].value_counts()),
            "artist_split_counts": dict(artist_splits["cold_start_split"].value_counts()),
        }

    def prepare_interactions(self, *, user_buckets: int = 32, chunk_size: int = 1_000_000) -> None:
        """Create time-respecting train/validation/test partitions by user bucket.

        The source has 252M events, so it is first converted to user-hash
        buckets.  Each bucket is then sorted only locally, avoiding a global
        in-memory sort while retaining full per-user chronology.
        """
        root = self.output / "interactions"
        bucket_root = root / "events_by_user_bucket"
        split_root = root / "temporal_split"
        users_target = root / "users.parquet"
        if not self.require_new(bucket_root):
            return
        bucket_root.mkdir(parents=True, exist_ok=True)
        source = self.data / "music4all_onion" / "userid_trackid_timestamp.tsv.bz2"
        counts: Counter[str] = Counter()
        rows_read = 0
        invalid_timestamps = 0
        for chunk_number, chunk in enumerate(pd.read_csv(source, sep="\t", dtype={"user_id": "string", "track_id": "string"}, chunksize=chunk_size), start=1):
            timestamps = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            valid = timestamps.notna()
            invalid_timestamps += int((~valid).sum())
            chunk = chunk.loc[valid, ["user_id", "track_id"]].copy()
            chunk["timestamp"] = timestamps.loc[valid].astype("int64") // 1_000_000_000
            chunk["user_bucket"] = chunk["user_id"].map(lambda value: stable_bucket(str(value), user_buckets)).astype("int16")
            counts.update(chunk["user_id"].tolist())
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            ds.write_dataset(
                table,
                base_dir=bucket_root,
                format="parquet",
                partitioning=ds.partitioning(pa.schema([pa.field("user_bucket", pa.int16())]), flavor="hive"),
                basename_template=f"events-{chunk_number:05d}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
            )
            rows_read += len(chunk)
            print(f"[interactions] chunk={chunk_number} valid_rows={rows_read:,} users={len(counts):,}", flush=True)
        users = pd.DataFrame({"user_id": list(counts), "event_count": list(counts.values())})
        users["eligible_min_10"] = users["event_count"] >= 10
        users["user_bucket"] = users["user_id"].map(lambda value: stable_bucket(value, user_buckets)).astype(np.int16)
        users.to_parquet(users_target, index=False)
        eligible_users = set(users.loc[users["eligible_min_10"], "user_id"])

        if split_root.exists():
            if not self.overwrite:
                raise FileExistsError(f"Temporal split already exists: {split_root}. Use --overwrite to rebuild it.")
            resolved = split_root.resolve()
            if root.resolve() not in resolved.parents:
                raise RuntimeError(f"Refusing to delete outside interactions output: {resolved}")
            shutil.rmtree(resolved)
        split_counts: Counter[str] = Counter()
        split_root.mkdir(parents=True, exist_ok=True)
        for bucket in range(user_buckets):
            bucket_dir = bucket_root / f"user_bucket={bucket}"
            files = sorted(bucket_dir.glob("*.parquet")) if bucket_dir.exists() else []
            if not files:
                continue
            frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
            frame = frame.loc[frame["user_id"].isin(eligible_users), ["user_id", "track_id", "timestamp"]]
            frame = frame.sort_values(["user_id", "timestamp", "track_id"], kind="stable").reset_index(drop=True)
            position = frame.groupby("user_id", sort=False).cumcount()
            total = frame.groupby("user_id", sort=False)["user_id"].transform("size")
            train_end = np.floor(total * 0.70).astype(np.int64)
            validation_end = np.floor(total * 0.80).astype(np.int64)
            frame["split"] = np.where(position < train_end, "train", np.where(position < validation_end, "validation", "test"))
            frame["event_position"] = position.astype(np.int32)
            frame["events_for_user"] = total.astype(np.int32)
            for split in ("train", "validation", "test"):
                output_dir = split_root / f"split={split}" / f"user_bucket={bucket}"
                output_dir.mkdir(parents=True, exist_ok=True)
                selected = frame.loc[frame["split"] == split]
                selected.to_parquet(output_dir / "events.parquet", index=False, compression="zstd")
                split_counts[split] += len(selected)
            print(f"[temporal-split] bucket={bucket} rows={len(frame):,}", flush=True)
        write_json(root / "temporal_split_manifest.json", {
            "seed": SEED, "min_user_interactions": 10, "split_rule": "per-user temporal 70/10/20",
            "user_buckets": user_buckets, "events_read": rows_read, "invalid_timestamps": invalid_timestamps,
            "eligible_users": len(eligible_users), "split_rows": dict(split_counts),
        })
        self.stats["stages"]["interactions"] = {"events_read": rows_read, "invalid_timestamps": invalid_timestamps, "users": len(users), "eligible_users": len(eligible_users), "split_rows": dict(split_counts)}

    def write_manifest(self) -> None:
        source_files = [
            self.data / "music4all_onion" / "userid_trackid_timestamp.tsv.bz2",
            self.data / "music4all_onion" / "id_essentia.tsv.bz2",
            self.data / "music4all_onion" / "id_lyrics_tf-idf.tsv.bz2",
            self.data / "music4all_onion" / "id_resnet.tsv.bz2",
            self.data / "music4all_onion" / "id_tags_dict.tsv.bz2",
            self.data / "music4all_onion" / "processed_lyrics.tar.gz",
            self.data / "music4all_aa" / "artist_modality_splits.json",
            self.data / "music4all_aa" / "album_modality_splits.json",
            self.data / "musicsem" / "train.csv",
            self.data / "fma" / "fma_metadata" / "fma_metadata" / "tracks.csv",
        ]
        reconstructed: dict[str, Any] = {}
        entities = self.output / "entities"
        if (entities / "aa_artists.parquet").exists():
            reconstructed["music4all_aa"] = {
                "artists": len(pd.read_parquet(entities / "aa_artists.parquet", columns=["artist_id"])),
                "albums": len(pd.read_parquet(entities / "aa_albums.parquet", columns=["album_id"])),
                "artist_track_edges": len(pd.read_parquet(entities / "aa_track_artists.parquet", columns=["track_id"])),
                "album_track_edges": len(pd.read_parquet(entities / "aa_track_albums.parquet", columns=["track_id"])),
            }
        social_path = self.output / "social" / "musicsem_social_semantics.parquet"
        if social_path.exists():
            social = pd.read_parquet(social_path, columns=["thread_id", "thread_split", "song_normalized", "artist_normalized"])
            reconstructed["musicsem"] = {
                "rows": len(social), "unique_threads": int(social["thread_id"].nunique()),
                "unique_songs": int(social["song_normalized"].nunique(dropna=True)),
                "unique_artists": int(social["artist_normalized"].nunique(dropna=True)),
                "split_counts": dict(social["thread_split"].dropna().value_counts()),
            }
        fma_path = self.output / "external" / "fma_tracks.parquet"
        if fma_path.exists():
            fma = pd.read_parquet(fma_path, columns=["track_id", "artist_id", "audio_available"])
            reconstructed["fma"] = {"tracks": len(fma), "artists": int(fma["artist_id"].nunique(dropna=True)), "audio_available": int(fma["audio_available"].sum())}
        modality_summary: dict[str, Any] = {}
        for name in ("audio_essentia", "lyrics_tfidf", "visual_resnet"):
            metadata = self.output / "features" / "onion" / name / "metadata.json"
            if metadata.exists():
                payload = read_json(metadata)
                modality_summary[name] = {
                    "rows": payload["rows"], "dimension": payload["dimension"], "dtype": payload["dtype"],
                    "malformed_rows": payload["malformed_rows"],
                    "shards": sum(1 for part in metadata.parent.glob("part-*.npy") if not part.name.endswith(".track_ids.npy")),
                }
        if modality_summary:
            reconstructed["onion_dense_features"] = modality_summary
        tracks_path = entities / "onion_tracks.parquet"
        tag_matrix_path = self.output / "features" / "onion" / "tag_weights" / "matrix.npz"
        if tracks_path.exists() and tag_matrix_path.exists():
            tracks = pd.read_parquet(tracks_path, columns=["track_id", "has_audio", "has_lyrics_feature", "has_visual_feature"])
            matrix = sparse.load_npz(tag_matrix_path)
            reconstructed["onion_tags_tracks"] = {
                "tracks": len(tracks), "tag_vocabulary": int(matrix.shape[1]), "nonzero_tag_edges": int(matrix.nnz),
                "audio_available": int(tracks["has_audio"].sum()), "lyrics_available": int(tracks["has_lyrics_feature"].sum()),
                "visual_available": int(tracks["has_visual_feature"].sum()),
            }
        alignment_manifest = self.output / "alignments" / "alignment_manifest.json"
        if alignment_manifest.exists():
            reconstructed["entity_alignment"] = read_json(alignment_manifest)
        interaction_manifest = self.output / "interactions" / "temporal_split_manifest.json"
        if interaction_manifest.exists():
            reconstructed["interactions"] = read_json(interaction_manifest)
        existing_manifest = self.output / "data_manifest.json"
        existing_stages = read_json(existing_manifest).get("stages", {}) if existing_manifest.exists() else {}
        self.stats["stages"] = {**existing_stages, **reconstructed, **self.stats["stages"]}
        self.stats["source_files"] = [source_file_record(path, self.data, with_sha256=self.checksum) for path in source_files if path.exists()]
        write_json(self.output / "data_manifest.json", self.stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stages", nargs="+", default=["all"],
        choices=["all", "extract", "aa", "musicsem", "fma", "features", "tags", "align", "interactions", "manifest"],
        help="Stages to run. 'all' includes interactions, which processes the full 252M-event log.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs for requested stages.")
    parser.add_argument("--checksum", action="store_true", help="Compute SHA256 for core source files in the manifest.")
    parser.add_argument("--interaction-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--user-buckets", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args.project_root, overwrite=args.overwrite, checksum=args.checksum)
    stages = set(args.stages)
    if "all" in stages:
        stages = {"extract", "aa", "musicsem", "fma", "features", "tags", "align", "interactions", "manifest"}
    if "extract" in stages:
        pipeline.extract_lyrics()
    if "aa" in stages:
        pipeline.prepare_music4all_aa()
    if "musicsem" in stages:
        pipeline.prepare_musicsem()
    if "fma" in stages:
        pipeline.prepare_fma()
    if "features" in stages:
        pipeline.prepare_onion_features()
    if "tags" in stages:
        pipeline.prepare_onion_tags_and_tracks()
    if "align" in stages:
        pipeline.prepare_entity_alignment_and_splits()
    if "interactions" in stages:
        pipeline.prepare_interactions(user_buckets=args.user_buckets, chunk_size=args.interaction_chunk_size)
    if "manifest" in stages or stages:
        pipeline.write_manifest()
    print(json.dumps(pipeline.stats, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
