"""Build a Mitsuba scene from a CJSON molecule and render it to a PNG."""

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .backend import select_variant
from .camera import camera_from_cjson
from .geometry import tessellate
from .scene import build_scene

try:  # progress reporting; silent unless Avogadro asked for it
    from avogadro.command import report_progress
except ImportError:  # pragma: no cover - avogadro wheel not installed
    def report_progress(message=None, value=None, maximum=None):
        return None

MAX_PASSES = 8
MAX_DEPTH = 8
# a mask surface that lets the ray through still consumes one bounce
MAX_DEPTH_TRANSLUCENT = 32

# "Material" option -> BSDF description (the base_color is added per mesh)
MATERIALS = {
    "Glossy plastic": {"type": "principled", "roughness": 0.3, "specular": 0.5},
    "Matte (clay)": {"type": "diffuse"},
    "Metallic": {"type": "principled", "metallic": 1.0, "roughness": 0.25},
}

# Studio lighting: a dim environment fill plus a soft key light. The background
# is composited afterwards (the environment is hidden from the camera), so the
# fill can be dim without graying the background.
FILL_RADIANCE = 0.4
KEY_DIRECTION = (0.7, 0.8, 1.0)  # (camera-left, camera-up, toward-camera)
KEY_DISTANCE = 3.0  # in bounding radii
KEY_SIZE = 1.5  # light edge, in bounding radii
KEY_IRRADIANCE = 3.0


@dataclass
class RenderResult:
    path: str
    variant: str
    width: int
    height: int
    seconds: float
    warnings: list = field(default_factory=list)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _make_mesh(mi, name, data, bsdf):
    """Create an mi.Mesh from numpy arrays using the variant's array types."""
    mesh = mi.Mesh(name, vertex_count=len(data.vertices),
                   face_count=len(data.faces), has_vertex_normals=True,
                   props=mi.Properties())
    params = mi.traverse(mesh)
    # the parameter types differ by variant (scalar vs JIT), so reuse them
    for key, array in (("vertex_positions", data.vertices),
                       ("vertex_normals", data.normals),
                       ("faces", data.faces)):
        params[key] = type(params[key])(np.ascontiguousarray(array).ravel())
    params.update()
    mesh.set_bsdf(bsdf)
    return mesh


def _lighting(scene_dict, camera, centre, radius, transparent):
    """Add the studio environment fill and the soft key light."""
    scene_dict["fill"] = {"type": "constant",
                          "radiance": {"type": "rgb", "value": [FILL_RADIANCE] * 3}}
    # Mitsuba camera axes in world space: +X is image-left, +Y up, +Z forward
    left, up, forward = (camera.to_world[:3, i] for i in range(3))
    direction = KEY_DIRECTION[0] * left + KEY_DIRECTION[1] * up - KEY_DIRECTION[2] * forward
    direction /= np.linalg.norm(direction)
    distance = KEY_DISTANCE * radius
    size = KEY_SIZE * radius
    radiance = KEY_IRRADIANCE * distance**2 / size**2
    import mitsuba as mi

    scene_dict["key"] = {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f().look_at(
            origin=(centre + direction * distance).tolist(),
            target=centre.tolist(), up=up.tolist(),
        ) @ mi.ScalarTransform4f().scale([size / 2, size / 2, 1]),
        "emitter": {"type": "area",
                    "radiance": {"type": "rgb", "value": [radiance] * 3}},
        "bsdf": {"type": "diffuse",
                 "reflectance": {"type": "rgb", "value": [0.0] * 3}},
    }


def build_mitsuba_scene(mi, scene, camera, width, height, options):
    """Assemble the Mitsuba scene dictionary and load it."""
    meshes = tessellate(scene, options.get("detail", "Medium"))
    material = MATERIALS.get(options.get("material"), MATERIALS["Glossy plastic"])
    transparent = bool(options.get("transparent", False))

    sensor = camera.sensor_dict(width, height)
    sensor["to_world"] = mi.ScalarTransform4f(sensor["to_world"].tolist())
    sensor["sampler"] = {"type": "independent", "sample_count": 1}
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path",
                       "max_depth": (MAX_DEPTH_TRANSLUCENT if scene.translucent
                                     else MAX_DEPTH),
                       "hide_emitters": True},
        "sensor": sensor,
    }

    points = np.concatenate([scene.sphere_centers, scene.cylinder_starts,
                             scene.cylinder_ends])
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = np.linalg.norm(points - centre, axis=1).max() + 1.0
    _lighting(scene_dict, camera, centre, radius, transparent)

    for i, (key, data) in enumerate(meshes.items()):
        linear = srgb_to_linear(np.array(key[:3]) / 255.0).tolist()
        bsdf_dict = dict(material)
        bsdf_dict["base_color" if material["type"] == "principled"
                  else "reflectance"] = {"type": "rgb", "value": linear}
        opacity = key[3] / 255.0
        if opacity < 1.0:  # alpha blending, as in Avogadro's translucent layers
            bsdf_dict = {"type": "mask",
                         "opacity": {"type": "rgb", "value": [opacity] * 3},
                         "bsdf": bsdf_dict}
        bsdf = mi.load_dict(bsdf_dict)
        scene_dict[f"mesh{i}"] = _make_mesh(mi, f"mesh{i}", data, bsdf)
    return mi.load_dict(scene_dict)


def render_image(mi, mscene, samples, progress=True):
    """Render in several passes, averaging; returns a (h, w, 4) float array."""
    passes = max(1, min(MAX_PASSES, samples))
    counts = [samples // passes + (1 if i < samples % passes else 0)
              for i in range(passes)]
    total = np.zeros(0)
    for i, spp in enumerate(counts):
        image = np.array(mi.render(mscene, spp=spp, seed=i), dtype=np.float64)
        total = image * spp if i == 0 else total + image * spp
        if progress:
            report_progress(f"Rendering pass {i + 1} of {passes}", i + 1, passes)
    return (total / samples).astype(np.float32)


def write_png(mi, path, image, transparent, background=(1.0, 1.0, 1.0)):
    """Write an 8-bit sRGB PNG. The film is alpha-premultiplied: either
    un-premultiply it (transparent) or composite it over the background."""
    rgb, alpha = image[..., :3], image[..., 3:4]
    if transparent:
        rgb = np.where(alpha > 1e-6, rgb / np.maximum(alpha, 1e-6), 0.0)
        out = np.concatenate([np.clip(rgb, 0, 1), alpha], axis=-1)
    else:
        out = rgb + (1.0 - alpha) * srgb_to_linear(background)
    # (write_bitmap would drop the alpha channel, so convert explicitly)
    fmt = mi.Bitmap.PixelFormat.RGBA if transparent else mi.Bitmap.PixelFormat.RGB
    bitmap = mi.Bitmap(np.ascontiguousarray(out, dtype=np.float32), fmt)
    bitmap.convert(pixel_format=fmt, component_format=mi.Struct.Type.UInt8,
                   srgb_gamma=True).write(path)


def unique_path(directory, stem, extension):
    """A path in directory that does not overwrite an existing render."""
    path = os.path.join(directory, stem + extension)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{counter}{extension}")
        counter += 1
    return path


def run_render(avo_input):
    """Render the CJSON molecule; returns Avogadro's {"message"}/{"error"}."""
    options = avo_input.get("options", {})
    cjson = avo_input.get("cjson", {})
    scene = build_scene(cjson)
    if scene.sphere_count == 0 and scene.cylinder_count == 0:
        return {"error": "No molecule data received. Please open a molecule first."}

    output_dir = os.path.expanduser(options.get("output_dir", "~/mitsuba-output"))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        return {"error": f"Could not create the output directory: {exc}"}
    name = str(cjson.get("name", "")).strip() or "molecule"
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    try:
        result = render_scene(scene, cjson, options, output_dir, name)
    except Exception as exc:
        return {"error": f"Mitsuba failed to render the molecule: {exc}"}

    lines = [f"Rendered image ({result.width} x {result.height}, "
             f"{result.variant}, {result.seconds:.1f} s) written to:",
             "", result.path]
    if result.warnings:
        lines += ["", "Notes:"] + [f"- {w}" for w in result.warnings]
    return {"message": "\n".join(lines)}


def render_scene(scene, cjson, options, output_dir, name):
    """Render a built Scene to a PNG in output_dir."""
    start = time.time()
    warnings = list(scene.warnings)
    tried = []
    variant = select_variant(options.get("backend", "Auto"), tried)
    import mitsuba as mi

    report_progress(f"Building the scene ({variant})")
    camera = camera_from_cjson(cjson, scene.positions)
    width = int(options.get("width", 1200))
    height = camera.image_height(width)
    transparent = bool(options.get("transparent", False))

    mscene = build_mitsuba_scene(mi, scene, camera, width, height, options)
    image = render_image(mi, mscene, int(options.get("samples", 128)))

    path = unique_path(output_dir, name, ".png")
    write_png(mi, path, image, transparent)
    return RenderResult(path, variant, width, height, time.time() - start, warnings)
