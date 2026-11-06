import numpy as np
import pytest

from avogadro_mitsuba.elements import ELEMENT_COLOR, ELEMENT_VDW
from avogadro_mitsuba.scene import (build_scene, parse_ball_and_stick,
                                    parse_opacity)

from conftest import molecule


def distance_from_axis(point, p0, p1):
    axis = (p1 - p0) / np.linalg.norm(p1 - p0)
    v = point - p0
    return np.linalg.norm(v - np.dot(v, axis) * axis)


def test_element_tables():
    assert len(ELEMENT_VDW) == len(ELEMENT_COLOR) == 119
    assert ELEMENT_VDW[6] == pytest.approx(1.77)
    assert ELEMENT_VDW[1] == pytest.approx(1.2)
    assert all(len(c) == 3 for c in ELEMENT_COLOR)


def test_settings_parsing():
    s = parse_ball_and_stick("true false 0.500000 0.200000 0.360000")
    assert (s.multi_bonds, s.show_hydrogens) == (True, False)
    assert (s.atom_scale, s.bond_radius, s.opacity) == (0.5, 0.2, 0.36)
    # four tokens (no opacity), clamping, and comma decimals
    s = parse_ball_and_stick("false true 5.0 0,3")
    assert (s.multi_bonds, s.atom_scale, s.bond_radius, s.opacity) == (False, 0.9, 0.3, 1.0)
    assert parse_ball_and_stick(None).atom_scale == 0.3
    assert parse_opacity("0.5").opacity == 0.5


def test_caffeine_counts_and_radii(caffeine):
    scene = build_scene(caffeine)
    n_atoms = len(caffeine["atoms"]["elements"]["number"])
    numbers = np.array(caffeine["atoms"]["elements"]["number"])
    assert n_atoms == 24
    assert scene.sphere_count == n_atoms
    np.testing.assert_allclose(
        scene.sphere_radii, np.asarray(ELEMENT_VDW)[numbers] * 0.3)
    np.testing.assert_allclose(scene.sphere_colors[numbers == 6][0],
                               np.array(ELEMENT_COLOR[6]) / 255.0)
    # one cylinder per single bond, two per double; two-colour bonds are split
    orders = np.array(caffeine["bonds"]["order"])
    pairs = np.array(caffeine["bonds"]["connections"]["index"]).reshape(-1, 2)
    expected = 0
    for (a, b), order in zip(pairs, orders):
        pieces = 1 if numbers[a] == numbers[b] else 2
        expected += pieces * (2 if order == 2 else 1)
    assert scene.cylinder_count == expected
    assert scene.cylinder_radii.min() == pytest.approx(0.1)
    assert scene.cylinder_radii.max() == pytest.approx(0.13)
    assert not scene.warnings


def test_custom_colors_and_hidden_hydrogens(colors):
    atoms = colors["atoms"]
    numbers = np.array(atoms["elements"]["number"])
    scene = build_scene(colors)
    # settings: "true false 0.3 0.1 0.36" -> hydrogens hidden, opacity warning
    heavy = int((numbers != 1).sum())
    assert scene.sphere_count == heavy
    assert any("opacity" in w for w in scene.warnings)
    rgb = np.array(atoms["colors"], dtype=float).reshape(-1, 3) / 255.0
    np.testing.assert_allclose(scene.sphere_colors, rgb[numbers != 1])
    # no bond to a hydrogen survives: every cylinder end lies on a heavy atom
    positions = scene.positions
    hydrogens = positions[numbers == 1]
    for end in np.concatenate([scene.cylinder_starts, scene.cylinder_ends]):
        assert np.linalg.norm(hydrogens - end, axis=1).min() > 0.5


def test_bond_orders_make_parallel_cylinders():
    r = 0.1
    single = build_scene(molecule([6, 6], [(0, 0, 0), (0, 0, 1.4)], [(0, 1)], [1]))
    double = build_scene(molecule([6, 6], [(0, 0, 0), (0, 0, 1.4)], [(0, 1)], [2]))
    triple = build_scene(molecule([6, 6], [(0, 0, 0), (0, 0, 1.4)], [(0, 1)], [3]))
    assert (single.cylinder_count, double.cylinder_count, triple.cylinder_count) == (1, 2, 3)
    p0 = np.zeros(3)
    p1 = np.array([0, 0, 1.4])
    # double: +-r from the axis, radius 1.3 r
    for s, e in zip(double.cylinder_starts, double.cylinder_ends):
        assert distance_from_axis(s, p0, p1) == pytest.approx(r)
    assert double.cylinder_radii == pytest.approx([1.3 * r] * 2)
    # triple: centre bond plus two at 2r with radius 1.15 r
    dist = sorted(distance_from_axis(s, p0, p1) for s in triple.cylinder_starts)
    assert dist == pytest.approx([0.0, 2 * r, 2 * r], abs=1e-9)
    assert sorted(triple.cylinder_radii) == pytest.approx([r, 1.15 * r, 1.15 * r])


def test_multi_bonds_off_draws_single_cylinder(clone):
    cj = molecule([6, 6], [(0, 0, 0), (0, 0, 1.4)], [(0, 1)], [2])
    cj["atoms"]["layer"] = [0, 0]
    cj["layer"] = {"visible": [True], "enable": {"Ball and Stick": [True]},
                   "settings": {"Ball and Stick": ["false true 0.3 0.1 1.0"]}}
    assert build_scene(cj).cylinder_count == 1


def test_two_colour_bond_splits_at_midpoint():
    scene = build_scene(molecule([6, 8], [(0, 0, 0), (0, 0, 2.0)], [(0, 1)], [1]))
    assert scene.cylinder_count == 2
    assert scene.cylinder_ends[0] == pytest.approx([0, 0, 1.0])
    assert scene.cylinder_starts[1] == pytest.approx([0, 0, 1.0])
    np.testing.assert_allclose(scene.cylinder_colors[0], np.array(ELEMENT_COLOR[6]) / 255)
    np.testing.assert_allclose(scene.cylinder_colors[1], np.array(ELEMENT_COLOR[8]) / 255)


def test_missing_layer_block_defaults_to_ball_and_stick():
    scene = build_scene(molecule([6, 1], [(0, 0, 0), (0, 0, 1.09)], [(0, 1)], [1]))
    assert scene.sphere_count == 2
    assert scene.sphere_radii == pytest.approx([1.77 * 0.3, 1.2 * 0.3])
    assert scene.cylinder_radii == pytest.approx([0.1, 0.1])


def licorice_cjson(cj, style):
    cj = {**cj, "atoms": {**cj["atoms"], "layer": [0] * len(cj["atoms"]["elements"]["number"])}}
    cj["layer"] = {"visible": [True], "enable": {style: [True]}, "settings": {}}
    return cj


def test_licorice_and_vdw(caffeine):
    lic = build_scene(licorice_cjson(caffeine, "Licorice"))
    assert lic.sphere_count == 24
    assert set(np.round(lic.sphere_radii, 6)) == {0.2}
    assert set(np.round(lic.cylinder_radii, 6)) == {0.2}
    # licorice draws single cylinders only (no multiple-bond offsets)
    assert lic.cylinder_count < 2 * len(caffeine["bonds"]["order"])

    vdw = build_scene(licorice_cjson(caffeine, "Van der Waals"))
    numbers = np.array(caffeine["atoms"]["elements"]["number"])
    assert vdw.cylinder_count == 0
    np.testing.assert_allclose(vdw.sphere_radii, np.asarray(ELEMENT_VDW)[numbers])


def test_hidden_layer_and_unsupported_style(caffeine, clone):
    cj = clone(caffeine)
    cj["layer"]["visible"] = [False]
    scene = build_scene(cj)
    assert scene.sphere_count == 0 or any("drawing ball and stick" in w for w in scene.warnings)

    cj = clone(caffeine)
    cj["layer"]["enable"]["Ball and Stick"] = [False]
    scene = build_scene(cj)
    assert scene.sphere_count == 24  # fell back to ball and stick
    assert scene.warnings


def test_mixed_layers(clone):
    # atom 0 in a Van der Waals layer, atom 1 in a ball-and-stick layer
    cj = molecule([6, 6], [(0, 0, 0), (5, 0, 0)])
    cj["atoms"]["layer"] = [0, 1]
    cj["layer"] = {"visible": [True, True],
                   "enable": {"Van der Waals": [True, False],
                              "Ball and Stick": [False, True]},
                   "settings": {}}
    scene = build_scene(cj)
    assert sorted(scene.sphere_radii) == pytest.approx(sorted([1.77, 1.77 * 0.3]))
