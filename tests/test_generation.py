from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from coc_env.entities import Building
from coc_env.generation import (
    PRESET_LAYOUT_PROFILES,
    LayoutProfile,
    generate_layout,
    layout_quality_errors,
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
        assert layout_quality_errors(layout, profile) == []


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


def test_special_defenses_use_tactical_anchor_bands() -> None:
    profile = preset_layout_profile("hard")
    layout = generate_layout(profile, np.random.default_rng(7))
    townhall = next(b for b in layout if b.spec.kind == "townhall")
    storages = [b for b in layout if b.spec.kind == "storage"]
    cannons = [b for b in layout if b.spec.kind == "cannon"]
    wizard_towers = [b for b in layout if b.spec.kind == "wizard_tower"]
    mortars = [b for b in layout if b.spec.kind == "mortar"]
    bombs = [b for b in layout if b.spec.kind == "bomb"]
    walls = [b for b in layout if b.spec.kind == "wall"]

    def avg_distance_to_townhall(buildings: list[Building]) -> float:
        return sum(townhall.distance_to(b.center[0], b.center[1]) for b in buildings) / len(buildings)

    def avg_distance_to_value(buildings: list[Building]) -> float:
        targets = [townhall, *storages]
        return sum(
            min(target.distance_to(b.center[0], b.center[1]) for target in targets)
            for b in buildings
        ) / len(buildings)

    assert avg_distance_to_townhall(mortars) < avg_distance_to_townhall(cannons)
    assert avg_distance_to_value(wizard_towers) <= avg_distance_to_value(cannons)
    assert all(b.spec.hidden and b.spec.is_trap and not b.spec.counts_for_score for b in bombs)
    assert all(
        min(b.distance_to(w.center[0], w.center[1]) for w in walls) <= 4.0
        for b in bombs
    )


@pytest.mark.parametrize("name", ["medium", "hard", "split"])
def test_core_profiles_have_distributed_wall_compartments(name: str) -> None:
    profile = preset_layout_profile(name)
    layout = generate_layout(profile, np.random.default_rng(11))
    townhall = next(b for b in layout if b.spec.kind == "townhall")
    walls = {(b.x, b.y) for b in layout if b.spec.kind == "wall"}

    cx, cy = townhall.center
    assert any(y < townhall.y and abs(x + 0.5 - cx) <= 6.0 for x, y in walls)
    assert any(y > townhall.y + townhall.spec.size - 1 and abs(x + 0.5 - cx) <= 6.0 for x, y in walls)
    assert any(x < townhall.x and abs(y + 0.5 - cy) <= 6.0 for x, y in walls)
    assert any(x > townhall.x + townhall.spec.size - 1 and abs(y + 0.5 - cy) <= 6.0 for x, y in walls)

    far_walls = {
        cell for cell in walls
        if townhall.distance_to(cell[0] + 0.5, cell[1] + 0.5) > 7.0
    }
    assert len(far_walls) >= max(24, profile.wall_count // 3)
