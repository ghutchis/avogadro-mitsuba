"""Convert Avogadro's camera (OpenGL modelView / projection) to a Mitsuba sensor.

OpenGL eye space looks down -Z with +X to the right and +Y up. Mitsuba's
camera space looks down +Z with +X to the *left*, so the two differ by a half
turn about Y: diag(-1, 1, -1, 1). Pure numpy.
"""

from dataclasses import dataclass

import numpy as np

# OpenGL eye space -> Mitsuba camera space (rotation by pi about Y)
GL_TO_MITSUBA = np.diag([-1.0, 1.0, -1.0, 1.0])

DEFAULT_FOV = 30.0  # degrees, for the fallback camera
DEFAULT_ASPECT = 4.0 / 3.0


@dataclass
class Camera:
    to_world: np.ndarray  # 4x4 camera-to-world, including ortho scale
    orthographic: bool
    fov_y: float  # degrees, perspective only
    aspect: float  # image width / height

    def image_height(self, width):
        return max(1, int(round(width / self.aspect)))

    def sensor_dict(self, width, height=None):
        """Plain dict for mi.load_dict; `to_world` is still a numpy array
        (wrap it in mi.ScalarTransform4f)."""
        height = height or self.image_height(width)
        sensor = {
            "type": "orthographic" if self.orthographic else "perspective",
            "to_world": self.to_world,
            "near_clip": 0.01,
            "far_clip": 10000.0,
        }
        if not self.orthographic:
            sensor["fov"] = self.fov_y
            sensor["fov_axis"] = "y"
        sensor["film"] = {
            "type": "hdrfilm",
            "width": int(width),
            "height": int(height),
            "pixel_format": "rgba",
            "rfilter": {"type": "gaussian"},
        }
        return sensor


def _matrix(rows):
    try:
        m = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        return None
    return m if m.shape == (4, 4) and np.isfinite(m).all() else None


def from_view(model_view, projection):
    """Camera from Avogadro's modelView and projection matrices (nested rows,
    as in CJSON `properties`). Returns None if either is unusable."""
    mv, proj = _matrix(model_view), _matrix(projection)
    if mv is None or proj is None or proj[0, 0] == 0 or proj[1, 1] == 0:
        return None
    try:
        inverse = np.linalg.inv(mv)
    except np.linalg.LinAlgError:
        return None

    aspect = proj[1, 1] / proj[0, 0]  # P00 = P11 / aspect
    if aspect <= 0:
        return None
    orthographic = abs(proj[3, 3] - 1.0) < 1e-6 and abs(proj[3, 2]) < 1e-6
    to_world = inverse @ GL_TO_MITSUBA
    fov_y = 0.0
    if orthographic:
        # Mitsuba's window spans [-1, 1] on the longer film axis, so a single
        # uniform scale of max(half width, half height) reproduces the view.
        scale = max(1.0 / abs(proj[0, 0]), 1.0 / abs(proj[1, 1]))
        to_world = to_world @ np.diag([scale, scale, 1.0, 1.0])
    else:
        fov_y = float(np.degrees(2.0 * np.arctan(1.0 / abs(proj[1, 1]))))
    return Camera(to_world, orthographic, fov_y, float(aspect))


def fallback_view(positions, padding=1.5, aspect=DEFAULT_ASPECT):
    """Perspective camera looking at the centroid down the least-extended
    principal axis, with the most-extended axis horizontal."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(positions) == 0:
        positions = np.zeros((1, 3))
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    if len(positions) > 1:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes = np.zeros((3, 3))
        axes[: len(vt)] = vt
    else:
        axes = np.eye(3)
    right, up = axes[0], axes[1]
    if np.linalg.norm(up) < 1e-9:  # collinear: pick any perpendicular
        up = np.cross(right, [0.0, 0.0, 1.0])
        if np.linalg.norm(up) < 1e-9:
            up = np.cross(right, [0.0, 1.0, 0.0])
    up = up / np.linalg.norm(up)
    toward_viewer = np.cross(right, up)

    # distance at which every point (grown by `padding`) fits the frustum,
    # with a 5% margin; z is the offset toward the viewer
    tan_y = np.tan(np.radians(DEFAULT_FOV) / 2.0)
    tan_x = tan_y * aspect
    x, y, z = centered @ right, centered @ up, centered @ toward_viewer
    distance = np.max(np.maximum(z + (np.abs(x) + padding) / tan_x,
                                 z + (np.abs(y) + padding) / tan_y)) * 1.05

    # camera-to-world in OpenGL convention, then into Mitsuba's
    gl = np.eye(4)
    gl[:3, 0], gl[:3, 1], gl[:3, 2] = right, up, toward_viewer
    gl[:3, 3] = centroid + toward_viewer * distance
    return Camera(gl @ GL_TO_MITSUBA, False, DEFAULT_FOV, aspect)


def camera_from_cjson(cjson, positions):
    """The saved Avogadro view if present, else a PCA fallback."""
    props = cjson.get("properties", {})
    camera = from_view(props.get("modelView"), props.get("projection"))
    return camera or fallback_view(positions)
