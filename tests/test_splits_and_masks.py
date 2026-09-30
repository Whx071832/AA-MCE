"""Deterministic artist-disjoint splits and the shared fixed-seed availability masks."""

import hashlib
from collections import Counter

import numpy as np
import pytest

from src.data.prepare import stable_bucket
from src.experiments import extended_modalities, fma_cross_dataset
from src.experiments.optimized_multimodal_tagging import CONDITIONS
from src.experiments.robust_multimodal_tagging import MODALITIES, intervention, stable_int


def onion_split(artist_id: str) -> str:
    """The 8:1:1 rule of ``Pipeline.prepare_entity_alignment_and_splits`` (``align`` stage)."""
    return "train" if stable_bucket(str(artist_id), 10) < 8 else "validation" if stable_bucket(str(artist_id), 10) < 9 else "test"


# ---------------------------------------------------------------------------
# hashing and splits
# ---------------------------------------------------------------------------
def test_stable_int_is_an_8_byte_blake2b_digest():
    for text in ("", "artist-0001", "fma-artist-split/42", "音乐"):
        expected = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
        assert stable_int(text) == expected
    assert stable_int("artist-0001") == 6181370622128652875  # pinned: must never change


def test_stable_bucket_agrees_with_stable_int():
    for index in range(200):
        value = f"artist-{index}"
        assert stable_bucket(value, 10) == stable_int(value) % 10
        assert stable_bucket(value, 32) == stable_int(value) % 32


def test_onion_artist_split_is_deterministic_and_close_to_8_1_1():
    pinned = {"artist-0000": "validation", "artist-0001": "train", "artist-0003": "validation", "artist-0007": "test"}
    assert {artist: onion_split(artist) for artist in pinned} == pinned
    artists = [f"artist-{index:05d}" for index in range(20000)]
    first = [onion_split(artist) for artist in artists]
    assert first == [onion_split(artist) for artist in artists]
    share = {split: count / len(artists) for split, count in Counter(first).items()}
    assert share["train"] == pytest.approx(0.8, abs=0.01)
    assert share["validation"] == pytest.approx(0.1, abs=0.01)
    assert share["test"] == pytest.approx(0.1, abs=0.01)


def test_fma_artist_split_is_deterministic_and_close_to_8_1_1():
    pinned = {"1": "train", "9": "test", "10": "validation"}
    assert {artist: fma_cross_dataset.build_split(artist) for artist in pinned} == pinned
    artists = [str(index) for index in range(20000)]
    splits = [fma_cross_dataset.build_split(artist) for artist in artists]
    for artist, split in zip(artists[:500], splits[:500]):
        bucket = stable_int(f"fma-artist-split/{artist}") % 10
        assert split == ("train" if bucket <= 7 else "validation" if bucket == 8 else "test")
    share = Counter(splits)
    assert share["train"] / len(artists) == pytest.approx(0.8, abs=0.01)
    assert share["validation"] / len(artists) == pytest.approx(0.1, abs=0.01)


# ---------------------------------------------------------------------------
# availability conditions
# ---------------------------------------------------------------------------
def availability_matrix(seed: int = 0, tracks: int = 500, views: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = (rng.random((tracks, views)) < 0.85).astype(np.float32)
    matrix[:5] = 0.0
    matrix[:5, 0] = 1.0  # a few tracks with a single observed modality
    return matrix


def test_the_six_predeclared_conditions():
    assert CONDITIONS == ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
    assert MODALITIES == ("audio", "lyrics", "visual")


def test_masks_are_deterministic_per_condition_seed_and_do_not_modify_the_input():
    availability = availability_matrix()
    original = availability.copy()
    for number, condition in enumerate(CONDITIONS):
        first = intervention(condition, availability, 20260910 + number)
        second = intervention(condition, availability, 20260910 + number)
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(availability, original)


def test_deterministic_removals_and_random_conditions():
    availability = availability_matrix()
    available = availability.sum(axis=1)
    np.testing.assert_array_equal(intervention("observed", availability, 1), availability)
    no_audio = intervention("no_audio", availability, 1)
    assert not no_audio[:, 0].any()
    np.testing.assert_array_equal(no_audio[:, 1:], availability[:, 1:])
    drop_one = intervention("random_one_missing", availability, 20260914)
    keep_one = intervention("random_two_missing", availability, 20260915)
    for row in range(len(availability)):
        assert np.all(drop_one[row] <= availability[row]) and np.all(keep_one[row] <= availability[row])
        if available[row] > 1:
            assert drop_one[row].sum() == available[row] - 1  # "random drop one"
            assert keep_one[row].sum() == 1  # "keep only one"
        else:
            np.testing.assert_array_equal(drop_one[row], availability[row])
            np.testing.assert_array_equal(keep_one[row], availability[row])


def test_fma_uses_the_same_mask_protocol():
    availability = availability_matrix(seed=3)
    mapping = {"observed": "observed", "no_audio": "no_audio", "no_text": "no_lyrics", "no_social": "no_visual", "random_one_missing": "random_one_missing", "random_two_missing": "random_two_missing"}
    assert list(mapping) == fma_cross_dataset.CONDITIONS
    for number, (fma_condition, onion_condition) in enumerate(mapping.items()):
        np.testing.assert_array_equal(
            fma_cross_dataset.intervention(fma_condition, availability, 20260910 + number),
            intervention(onion_condition, availability, 20260910 + number),
        )


def test_six_view_conditions():
    senses = ["audio", "audio", "lyrics", "lyrics", "visual", "visual"]
    availability = availability_matrix(seed=5, views=6)
    available = availability.sum(axis=1)
    assert len(extended_modalities.CONDITIONS) == 7
    no_audio = extended_modalities.intervention("no_audio", availability, senses, 1)
    assert not no_audio[:, :2].any()
    np.testing.assert_array_equal(no_audio[:, 2:], availability[:, 2:])
    half = extended_modalities.intervention("random_half_views_missing", availability, senses, 20260915)
    keep = extended_modalities.intervention("keep_one_view", availability, senses, 20260916)
    for row in range(len(availability)):
        if available[row] > 1:
            assert half[row].sum() == available[row] - available[row] // 2
            assert keep[row].sum() == 1
        assert np.all(keep[row] <= availability[row])
