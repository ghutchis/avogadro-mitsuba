import numpy as np
import pytest

from avogadro_mitsuba.camera import (camera_from_cjson, fallback_view,
                                     from_view)
from avogadro_mitsuba.scene import build_scene


def test_identity_view_axes(colors):
    props = colors["properties"]
    cam = from_view(props["modelView"], props["projection"])
    assert not cam.orthographic
    # Mitsuba +Z (forward) is world -Z, +X (image left) is world -X, up is +Y
    forward = cam.to_world[:3, 2]
    assert forward == pytest.approx([0, 0, -1], abs=1e-6)
    assert cam.to_world[:3, 0] == pytest.approx([-1, 0, 0], abs=1e-6)
    assert cam.to_world[:3, 1] == pytest.approx([0, 1, 0], abs=1e-6)
    # the camera sits 13.32 A in front of the origin on +Z
    assert cam.to_world[:3, 3] == pytest.approx([0, 0, 13.32], abs=1e-3)
    assert cam.fov_y == pytest.approx(np.degrees(2 * np.arctan(1 / 2.0)))
    assert cam.aspect == pytest.approx(2.0 / 1.490114450454712)


def test_rotated_view_round_trip(aniline):
    props = aniline["properties"]
    mv = np.array(props["modelView"])
    cam = from_view(mv, props["projection"])
    # a world point maps to eye space via modelView; the camera-local
    # (Mitsuba) coordinates are that with x and z negated
    point = np.array([0.3, -1.2, 0.8, 1.0])
    eye = mv @ point
    local = np.linalg.inv(cam.to_world) @ point
    assert local == pytest.approx([-eye[0], eye[1], -eye[2], 1.0], abs=1e-4)
    # rotation part is orthonormal
    r = cam.to_world[:3, :3]
    assert r.T @ r == pytest.approx(np.eye(3), abs=1e-4)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-4)


def test_orthographic_scale_and_height():
    mv = np.eye(4)
    mv[2, 3] = -10
    proj = np.array([[0.25, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -0.1, -1], [0, 0, 0, 1]])
    cam = from_view(mv, proj)
    assert cam.orthographic
    # half width 4, half height 2: Mitsuba's [-1, 1] window on the long axis
    assert cam.to_world[0, 0] == pytest.approx(-4.0)
    assert cam.to_world[1, 1] == pytest.approx(4.0)
    assert cam.aspect == pytest.approx(2.0)
    assert cam.image_height(800) == 400
    sensor = cam.sensor_dict(800)
    assert sensor["type"] == "orthographic"
    assert sensor["film"]["height"] == 400


def test_perspective_sensor_dict(colors):
    props = colors["properties"]
    sensor = from_view(props["modelView"], props["projection"]).sensor_dict(600)
    assert sensor["type"] == "perspective"
    assert sensor["fov_axis"] == "y"
    assert sensor["fov"] == pytest.approx(53.13, abs=0.01)
    assert sensor["film"]["width"] == 600


def test_bad_matrices_return_none():
    assert from_view(None, None) is None
    assert from_view(np.zeros((4, 4)), np.zeros((4, 4))) is None
    assert from_view([[1, 2]], [[1]]) is None


def test_fallback_looks_at_molecule(caffeine):
    cam = camera_from_cjson(caffeine, build_scene(caffeine).positions)
    assert not cam.orthographic
    positions = np.array(caffeine["atoms"]["coords"]["3d"]).reshape(-1, 3)
    centroid = positions.mean(axis=0)
    origin = cam.to_world[:3, 3]
    forward = cam.to_world[:3, 2]
    to_centroid = (centroid - origin) / np.linalg.norm(centroid - origin)
    assert forward == pytest.approx(to_centroid, abs=1e-6)
    # caffeine is planar: the view direction is along the plane normal, and
    # every atom is inside the view frustum
    tan_y = np.tan(np.radians(cam.fov_y) / 2)
    local = (np.linalg.inv(cam.to_world) @ np.c_[positions, np.ones(len(positions))].T).T
    assert (local[:, 2] > 0).all()
    assert (np.abs(local[:, 1]) / local[:, 2] < tan_y).all()
    assert (np.abs(local[:, 0]) / local[:, 2] < tan_y * cam.aspect).all()


def test_fallback_single_atom_and_collinear():
    assert fallback_view([[0, 0, 0]]).to_world.shape == (4, 4)
    cam = fallback_view([[0, 0, 0], [0, 0, 2], [0, 0, 4]])
    assert np.isfinite(cam.to_world).all()
