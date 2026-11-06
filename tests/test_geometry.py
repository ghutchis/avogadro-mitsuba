import numpy as np
import pytest

from avogadro_mitsuba import geometry
from avogadro_mitsuba.scene import build_scene

from conftest import molecule


def face_normals(mesh):
    v = mesh.vertices[mesh.faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return n, v.mean(axis=1)


def test_icosphere_counts_and_unit_length():
    for level in range(4):
        v, f = geometry.icosphere(level)
        assert len(f) == 20 * 4**level
        assert len(v) == 10 * 4**level + 2
        assert np.linalg.norm(v, axis=1) == pytest.approx(1.0)


def test_sphere_winding_is_outward():
    mesh = geometry.sphere_meshes(np.array([[1.0, 2, 3]]), np.array([0.5]), 2)
    n, centres = face_normals(mesh)
    assert (np.einsum("ij,ij->i", n, centres - [1, 2, 3]) > 0).all()
    assert np.linalg.norm(mesh.vertices - [1, 2, 3], axis=1) == pytest.approx(0.5, abs=1e-6)


def test_cylinder_geometry():
    mesh = geometry.cylinder_meshes(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 2]]),
                                    np.array([0.3]), 16)
    assert len(mesh.vertices) == 32 and mesh.triangle_count == 32
    assert np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]) == pytest.approx(0.3, abs=1e-6)
    n, centres = face_normals(mesh)
    radial = centres.copy()
    radial[:, 2] = 0
    assert (np.einsum("ij,ij->i", n, radial) > 0).all()
    # normals are unit radial
    assert np.linalg.norm(mesh.normals, axis=1) == pytest.approx(1.0, abs=1e-6)


def test_degenerate_cylinder_is_dropped():
    mesh = geometry.cylinder_meshes(np.zeros((1, 3)), np.zeros((1, 3)), np.array([0.1]), 8)
    assert mesh.triangle_count == 0


def test_tessellate_merges_per_colour(caffeine):
    scene = build_scene(caffeine)
    meshes = geometry.tessellate(scene, "Low")
    # caffeine has H, C, N, O (cylinders share their atoms' colours)
    assert len(meshes) == 4
    total = sum(m.triangle_count for m in meshes.values())
    assert total == scene.sphere_count * 320 + scene.cylinder_count * 48
    for m in meshes.values():
        assert m.faces.max() < len(m.vertices)
        assert m.vertices.dtype == np.float32 and m.faces.dtype == np.uint32


def test_detail_is_lowered_for_huge_scenes(monkeypatch):
    monkeypatch.setattr(geometry, "TRIANGLE_BUDGET", 2000)
    scene = build_scene(molecule([6] * 5, [(i * 3.0, 0, 0) for i in range(5)]))
    meshes = geometry.tessellate(scene, "High")
    assert sum(m.triangle_count for m in meshes.values()) == 5 * 20 * 4**2


def test_tessellate_groups_by_opacity():
    cjson = molecule([6, 6], [(0, 0, 0), (1.5, 0, 0)], [(0, 1)], [1])
    cjson["layer"] = {
        "visible": [True],
        "enable": {"Licorice": [True], "Van der Waals": [True]},
        "settings": {"Licorice": ["1.0"], "Van der Waals": ["0.5"]},
    }
    meshes = geometry.tessellate(build_scene(cjson), "Low")
    # same colour, two opacities: opaque 255 and half 128
    assert sorted(key[3] for key in meshes) == [128, 255]
    assert len({key[:3] for key in meshes}) == 1
