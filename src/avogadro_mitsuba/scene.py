"""Convert Avogadro CJSON into renderer-independent primitives.

Pure numpy, no Mitsuba. The geometry is a port of the Ball and Stick,
Licorice and Van der Waals scene plugins in avogadrolibs, honouring the layer
visibility, enabled styles and per-layer settings stored in the CJSON.
"""

from dataclasses import dataclass, field

import numpy as np

from .elements import ELEMENT_COLOR, ELEMENT_VDW

BALL_AND_STICK = "Ball and Stick"
LICORICE = "Licorice"
VAN_DER_WAALS = "Van der Waals"
STYLES = (BALL_AND_STICK, LICORICE, VAN_DER_WAALS)

LICORICE_RADIUS = 0.2


@dataclass
class BallAndStickSettings:
    """Defaults match the plugin's QSettings defaults."""

    multi_bonds: bool = True
    show_hydrogens: bool = True
    atom_scale: float = 0.3
    bond_radius: float = 0.1
    opacity: float = 1.0


@dataclass
class OpacitySettings:
    opacity: float = 1.0


@dataclass
class Scene:
    """Spheres (center, radius, rgb, opacity) and cylinders (p0, p1, radius,
    rgb, opacity).

    Colors are sRGB in 0-1, opacities 0-1. `positions` holds every atom, drawn or not, for
    framing the camera.
    """

    sphere_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sphere_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sphere_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sphere_opacity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cylinder_starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cylinder_ends: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cylinder_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cylinder_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cylinder_opacity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    warnings: list = field(default_factory=list)

    @property
    def sphere_count(self):
        return len(self.sphere_radii)

    @property
    def cylinder_count(self):
        return len(self.cylinder_radii)

    @property
    def translucent(self):
        """Is any primitive drawn with opacity below 1?"""
        return bool((self.sphere_opacity < 1.0).any()
                    or (self.cylinder_opacity < 1.0).any())


def _to_bool(text):
    """Port of Core::stringToBool: 'true' / '1' are true."""
    return str(text).strip().lower() in ("true", "1")


def _to_float(token):
    """Parse a float, accepting a comma decimal separator; None if invalid."""
    try:
        value = float(str(token).replace(",", "."))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def parse_ball_and_stick(text):
    """Parse 'multiBonds showHydrogens atomScale bondRadius opacity'."""
    result = BallAndStickSettings()
    if not text:
        return result
    tokens = str(text).split()
    if len(tokens) > 0:
        result.multi_bonds = _to_bool(tokens[0])
    if len(tokens) > 1:
        result.show_hydrogens = _to_bool(tokens[1])
    if len(tokens) > 2:
        value = _to_float(tokens[2])
        if value is not None and value > 0:
            result.atom_scale = min(max(value, 0.1), 0.9)
    if len(tokens) > 3:
        value = _to_float(tokens[3])
        if value is not None and value > 0:
            result.bond_radius = min(max(value, 0.1), 0.8)
    if len(tokens) > 4:
        value = _to_float(tokens[4])
        if value is not None:
            result.opacity = min(max(value, 0.0), 1.0)
    return result


def parse_opacity(text):
    """Parse the single-opacity settings string of Licorice / Van der Waals."""
    result = OpacitySettings()
    tokens = str(text).split() if text else []
    if tokens:
        value = _to_float(tokens[0])
        if value is not None:
            result.opacity = min(max(value, 0.0), 1.0)
    return result


class _Layers:
    """Per-style view of the CJSON `layer` block."""

    def __init__(self, layer_block, atom_layers, atom_count):
        self.atom_layers = atom_layers
        self.atom_count = atom_count
        self.missing = not layer_block
        layer_block = layer_block or {}
        self.visible = layer_block.get("visible", [True])
        self.enable = layer_block.get("enable", {})
        self.settings = layer_block.get("settings", {})

    def layer_of(self, atom):
        if atom < len(self.atom_layers):
            return int(self.atom_layers[atom])
        return 0

    def is_enabled(self, style, layer):
        """Is the style drawn for this layer (enabled and visible)?"""
        if self.missing:
            return style == BALL_AND_STICK
        if layer < len(self.visible) and not self.visible[layer]:
            return False
        flags = self.enable.get(style, [])
        return layer < len(flags) and bool(flags[layer])

    def any_enabled(self, style):
        count = max(len(self.visible), 1)
        return any(self.is_enabled(style, i) for i in range(count))

    def setting_text(self, style, layer):
        entries = self.settings.get(style) or []
        if layer < len(entries):
            return entries[layer]
        return None


def atom_colors(atoms, numbers):
    """Per-atom sRGB colors in 0-1: custom `colors` if present, else element."""
    count = len(numbers)
    custom = atoms.get("colors")
    if custom:
        stride = len(custom) // count if count else 0
        if stride in (3, 4) and len(custom) == stride * count:
            rgb = np.asarray(custom, dtype=float).reshape(count, stride)[:, :3]
            return rgb / 255.0
    table = np.asarray(ELEMENT_COLOR, dtype=float) / 255.0
    return table[np.clip(numbers, 0, len(table) - 1)]


def _perpendicular(axis):
    """Port of Eigen's 3D unitOrthogonal()."""
    x, y, z = axis
    if abs(x) > abs(z):
        inv = 1.0 / np.hypot(x, z)
        return np.array([-z * inv, 0.0, x * inv])
    inv = 1.0 / np.hypot(y, z)
    return np.array([0.0, -z * inv, y * inv])


def _rotate(vector, axis, angle):
    """Rodrigues rotation of vector about the unit axis."""
    c, s = np.cos(angle), np.sin(angle)
    return (vector * c + np.cross(axis, vector) * s
            + axis * np.dot(axis, vector) * (1.0 - c))


def _bond_cylinders(p1, p2, order, radius, multi):
    """Cylinders (start, end, radius) for one bond, as in BallAndStick."""
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return []
    axis = axis / length
    effective = order if multi else 1
    if effective == 2:
        delta = _rotate(_perpendicular(axis), axis, np.pi / 4) * radius
        return [(p1 + delta, p2 + delta, radius * 1.3),
                (p1 - delta, p2 - delta, radius * 1.3)]
    cylinders = [(p1, p2, radius)]
    if effective == 3:
        delta = _rotate(_perpendicular(axis), axis, np.pi / 4) * 2.0 * radius
        cylinders += [(p1 + delta, p2 + delta, radius * 1.15),
                      (p1 - delta, p2 - delta, radius * 1.15)]
    return cylinders


class _Builder:
    def __init__(self):
        self.sphere = []
        self.cylinder = []

    def add_sphere(self, center, radius, rgb, opacity=1.0):
        self.sphere.append((center, radius, rgb, opacity))

    def add_cylinder(self, p0, p1, radius, rgb0, rgb1, opacity=1.0):
        """Add a cylinder, split at the midpoint when the colors differ."""
        if np.allclose(rgb0, rgb1):
            self.cylinder.append((p0, p1, radius, rgb0, opacity))
            return
        mid = 0.5 * (p0 + p1)
        self.cylinder.append((p0, mid, radius, rgb0, opacity))
        self.cylinder.append((mid, p1, radius, rgb1, opacity))

    def finish(self, scene):
        if self.sphere:
            scene.sphere_centers = np.array([s[0] for s in self.sphere])
            scene.sphere_radii = np.array([s[1] for s in self.sphere])
            scene.sphere_colors = np.array([s[2] for s in self.sphere])
            scene.sphere_opacity = np.array([s[3] for s in self.sphere])
        if self.cylinder:
            scene.cylinder_starts = np.array([c[0] for c in self.cylinder])
            scene.cylinder_ends = np.array([c[1] for c in self.cylinder])
            scene.cylinder_radii = np.array([c[2] for c in self.cylinder])
            scene.cylinder_colors = np.array([c[3] for c in self.cylinder])
            scene.cylinder_opacity = np.array([c[4] for c in self.cylinder])


def build_scene(cjson):
    """Build a Scene from a CJSON dictionary."""
    atoms = cjson.get("atoms", {})
    numbers = np.asarray(atoms.get("elements", {}).get("number", []), dtype=int)
    count = len(numbers)
    scene = Scene()
    if count == 0:
        return scene

    coords = np.asarray(atoms.get("coords", {}).get("3d", []), dtype=float)
    if coords.size != 3 * count:
        scene.warnings.append("No 3D coordinates found.")
        return scene
    positions = coords.reshape(count, 3)
    scene.positions = positions
    colors = atom_colors(atoms, numbers)
    radii_vdw = np.asarray(ELEMENT_VDW)[np.clip(numbers, 0, len(ELEMENT_VDW) - 1)]

    bonds = cjson.get("bonds", {})
    pairs = np.asarray(bonds.get("connections", {}).get("index", []), dtype=int)
    pairs = pairs.reshape(-1, 2) if pairs.size % 2 == 0 else np.zeros((0, 2), int)
    orders = list(bonds.get("order", []))
    orders += [1] * (len(pairs) - len(orders))

    layers = _Layers(cjson.get("layer"), atoms.get("layer", []), count)
    styles = [s for s in STYLES if layers.any_enabled(s)]
    if not styles:
        scene.warnings.append(
            "No Ball and Stick, Licorice or Van der Waals layer is enabled "
            "(other styles are not supported); drawing ball and stick."
        )
        layers = _Layers(None, [], count)
        styles = [BALL_AND_STICK]

    builder = _Builder()
    for style in styles:
        _add_style(builder, style, layers, positions, numbers, colors,
                   radii_vdw, pairs, orders)
    builder.finish(scene)
    return scene


def _add_style(builder, style, layers, positions, numbers, colors, radii_vdw,
               pairs, orders):
    """Add the geometry of one style across all layers."""
    parse = parse_ball_and_stick if style == BALL_AND_STICK else parse_opacity
    cache = {}

    def setting(layer):
        if layer not in cache:
            cache[layer] = parse(layers.setting_text(style, layer))
        return cache[layer]

    for i in range(len(numbers)):
        layer = layers.layer_of(i)
        if not layers.is_enabled(style, layer):
            continue
        conf = setting(layer)
        if style == BALL_AND_STICK:
            if numbers[i] == 1 and not conf.show_hydrogens:
                continue
            radius = radii_vdw[i] * conf.atom_scale
        elif style == LICORICE:
            radius = LICORICE_RADIUS
        else:
            radius = radii_vdw[i]
        builder.add_sphere(positions[i], radius, colors[i], conf.opacity)

    if style == VAN_DER_WAALS:
        return

    for (a1, a2), order in zip(pairs, orders):
        l1, l2 = layers.layer_of(a1), layers.layer_of(a2)
        if not (layers.is_enabled(style, l1) or layers.is_enabled(style, l2)):
            continue
        c1, c2 = setting(l1), setting(l2)
        # a bond between layers is as opaque as the most opaque enabled end
        opacity = max(c.opacity for c, l in ((c1, l1), (c2, l2))
                      if layers.is_enabled(style, l))
        p1, p2 = positions[a1], positions[a2]
        if style == LICORICE:
            builder.add_cylinder(p1, p2, LICORICE_RADIUS, colors[a1], colors[a2],
                                opacity)
            continue
        has_h = numbers[a1] == 1 or numbers[a2] == 1
        if has_h and not c1.show_hydrogens and not c2.show_hydrogens:
            continue
        radius = 0.5 * (c1.bond_radius + c2.bond_radius)
        multi = c1.multi_bonds or c2.multi_bonds
        for start, end, r in _bond_cylinders(p1, p2, int(order), radius, multi):
            builder.add_cylinder(start, end, r, colors[a1], colors[a2], opacity)
