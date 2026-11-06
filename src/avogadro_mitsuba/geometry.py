"""Tessellate scene primitives and merge them into one mesh per colour.

Mitsuba is slow to load one shape per atom (minutes on JIT variants), so all
spheres and cylinders of a colour become a single indexed triangle mesh.
Pure numpy.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# "Geometry Detail" option -> (icosphere subdivisions, cylinder sides)
DETAIL_LEVELS = {"Low": (2, 24), "Medium": (3, 32), "High": (4, 48)}

# Keep the merged triangle count below this by lowering the detail
TRIANGLE_BUDGET = 6_000_000


@dataclass
class MeshData:
    """An indexed triangle mesh with smooth vertex normals."""

    vertices: np.ndarray  # (n, 3) float32
    normals: np.ndarray  # (n, 3) float32
    faces: np.ndarray  # (m, 3) uint32

    @property
    def triangle_count(self):
        return len(self.faces)


@lru_cache(maxsize=None)
def icosphere(subdivisions):
    """Unit icosphere as (vertices, faces); faces wind counter-clockwise
    seen from outside."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        cache = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces, dtype=np.uint32)


def sphere_meshes(centers, radii, subdivisions):
    """One merged mesh for many spheres."""
    unit_v, unit_f = icosphere(subdivisions)
    n = len(radii)
    vertices = (unit_v[None, :, :] * radii[:, None, None]
                + centers[:, None, :]).reshape(-1, 3)
    normals = np.tile(unit_v, (n, 1))
    offsets = (np.arange(n, dtype=np.uint32) * len(unit_v))[:, None, None]
    faces = (unit_f[None, :, :] + offsets).reshape(-1, 3)
    return MeshData(vertices.astype(np.float32), normals.astype(np.float32),
                    faces.astype(np.uint32))


def _frames(axes):
    """Orthonormal (u, v) perpendicular to each unit axis, shape (n, 3)."""
    helper = np.where(np.abs(axes[:, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = np.cross(axes, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, np.cross(axes, u)


def cylinder_meshes(starts, ends, radii, sides):
    """One merged mesh for many open-ended cylinders."""
    n = len(radii)
    delta = ends - starts
    length = np.linalg.norm(delta, axis=1)
    keep = length > 1e-9
    starts, delta, radii, length = starts[keep], delta[keep], radii[keep], length[keep]
    n = len(radii)
    axes = delta / length[:, None]
    u, v = _frames(axes)
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    # radial direction per cylinder and angle: (n, sides, 3)
    radial = (np.cos(theta)[None, :, None] * u[:, None, :]
              + np.sin(theta)[None, :, None] * v[:, None, :])
    bottom = starts[:, None, :] + radii[:, None, None] * radial
    top = bottom + delta[:, None, :]
    vertices = np.concatenate([bottom, top], axis=1).reshape(-1, 3)
    normals = np.concatenate([radial, radial], axis=1).reshape(-1, 3)
    j = np.arange(sides)
    k = (j + 1) % sides
    # quad (b_j, b_k, t_k, t_j) split in two, outward-facing
    quad = np.concatenate([np.stack([j, k, sides + k], 1),
                           np.stack([j, sides + k, sides + j], 1)]).astype(np.uint32)
    offsets = (np.arange(n, dtype=np.uint32) * 2 * sides)[:, None, None]
    faces = (quad[None, :, :] + offsets).reshape(-1, 3)
    return MeshData(vertices.astype(np.float32), normals.astype(np.float32),
                    faces.astype(np.uint32))


def _merge(meshes):
    meshes = [m for m in meshes if m.triangle_count]
    if not meshes:
        return None
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return MeshData(
        np.concatenate([m.vertices for m in meshes]),
        np.concatenate([m.normals for m in meshes]),
        np.concatenate([m.faces + np.uint32(o) for m, o in zip(meshes, offsets)]),
    )


def _color_key(rgb, opacity=1.0):
    """(r, g, b, a) in 0-255: the grouping key of a mesh."""
    return tuple(int(round(c * 255)) for c in (*rgb, opacity))


def estimate_triangles(scene, subdivisions, sides):
    return scene.sphere_count * 20 * 4**subdivisions + scene.cylinder_count * 2 * sides


def tessellate(scene, detail="Medium"):
    """Merge a Scene into {(r, g, b, a) 0-255: MeshData}, one mesh per colour
    and opacity (sRGB colour, alpha = opacity).

    The detail is lowered if the mesh would exceed TRIANGLE_BUDGET.
    """
    subdivisions, sides = DETAIL_LEVELS.get(detail, DETAIL_LEVELS["Medium"])
    while subdivisions > 1 and estimate_triangles(scene, subdivisions, sides) > TRIANGLE_BUDGET:
        subdivisions -= 1
        sides = max(12, sides - 8)

    groups = {}
    for i, rgb in enumerate(scene.sphere_colors):
        key = _color_key(rgb, scene.sphere_opacity[i])
        groups.setdefault(key, ([], []))[0].append(i)
    for i, rgb in enumerate(scene.cylinder_colors):
        key = _color_key(rgb, scene.cylinder_opacity[i])
        groups.setdefault(key, ([], []))[1].append(i)

    result = {}
    for key, (sphere_ids, cylinder_ids) in groups.items():
        parts = []
        if sphere_ids:
            s = np.asarray(sphere_ids)
            parts.append(sphere_meshes(scene.sphere_centers[s],
                                       scene.sphere_radii[s], subdivisions))
        if cylinder_ids:
            c = np.asarray(cylinder_ids)
            parts.append(cylinder_meshes(scene.cylinder_starts[c],
                                         scene.cylinder_ends[c],
                                         scene.cylinder_radii[c], sides))
        merged = _merge(parts)
        if merged is not None:
            result[key] = merged
    return result
