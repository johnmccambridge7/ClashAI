from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from coc_env.generation import (
    PRESET_LAYOUT_PROFILES,
    LayoutProfile,
    generate_layout,
    preset_layout_profile,
    validate_layout,
)


def _fingerprint(profile: LayoutProfile, seed: int) -> list[tuple[str, int, int]]:
    layout = generate_layout(profile, np.random.default_rng(seed))
    return [(b.spec.kind, b.x, b.y) for b in layout]


@pytest.mark.parametrize("name", sorted(PRESET_LAYOUT_PROFILES))
def test_generated_layouts_are_valid(name: str) -> None:
    profile = preset_layout_profile(name)
    for seed in range(5):
        layout = generate_layout(profile, np.random.default_rng(seed))
        assert validate_layout(layout, profile) == []


def test_generated_layout_is_seed_deterministic() -> None:
    profile = preset_layout_profile("medium")
    assert _fingerprint(profile, 123) == _fingerprint(profile, 123)
    assert _fingerprint(profile, 123) != _fingerprint(profile, 124)


def test_generated_layout_matches_profile_counts() -> None:
    profile = preset_layout_profile("hard")
    layout = generate_layout(profile, np.random.default_rng(7))
    counts = Counter(b.spec.kind for b in layout)

    for kind, expected in profile.building_counts:
        assert counts[kind] == expected
    assert counts["wall"] == profile.wall_count
    assert len(layout) == profile.max_buildings
