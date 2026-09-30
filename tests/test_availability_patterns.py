"""Availability patterns: enumeration, naming and per-track lookup."""

import numpy as np
import pytest

from src.experiments import aa_core
from src.experiments import availability_aware_ensemble as aa3

VIEWS3 = ["audio", "lyrics", "visual"]
VIEWS6 = ["audio_essentia", "audio_ivec1024", "lyrics_tfidf", "lyrics_word2vec", "visual_resnet", "visual_incp"]


@pytest.mark.parametrize(("count", "expected"), [(1, 1), (2, 3), (3, 7), (6, 63)])
def test_all_non_empty_patterns_are_enumerated(count, expected):
    patterns = aa_core.all_patterns(count)
    assert len(patterns) == expected == 2**count - 1
    assert len(set(patterns)) == expected
    assert all(len(pattern) == count and sum(pattern) > 0 for pattern in patterns)
    assert tuple([0] * count) not in patterns
    assert patterns[0] == tuple([1] * count)  # the fully observed pattern comes first


def test_three_view_runner_uses_the_same_patterns_as_the_shared_core():
    assert aa3.PATTERNS == aa_core.all_patterns(3)
    assert [aa3.pattern_name(p) for p in aa3.PATTERNS] == [aa_core.pattern_name(p, VIEWS3) for p in aa3.PATTERNS]


def test_pattern_names():
    assert aa_core.pattern_name((1, 1, 1), VIEWS3) == "audio+lyrics+visual"
    assert aa_core.pattern_name((1, 0, 1), VIEWS3) == "audio+visual"
    assert aa3.pattern_name((0, 1, 0)) == "lyrics"
    six_view_names = [aa_core.pattern_name(p, VIEWS6) for p in aa_core.all_patterns(6)]
    assert len(set(six_view_names)) == 63


def test_pattern_index_maps_rows_to_patterns_with_fallback():
    patterns = aa_core.all_patterns(3)
    observed = patterns.index((1, 1, 1))
    availability = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    index = aa_core.pattern_index(availability, patterns, fallback=observed)
    assert [patterns[i] for i in index[:3]] == [(1, 1, 1), (0, 1, 1), (1, 0, 0)]
    assert index[3] == observed  # a track without any modality uses the observed pattern
    # The three-view runner marks such rows with -1 before applying the same fallback.
    expected = [observed, patterns.index((0, 1, 1)), patterns.index((1, 0, 0)), -1]
    np.testing.assert_array_equal(aa3.pattern_index(availability), np.asarray(expected))
