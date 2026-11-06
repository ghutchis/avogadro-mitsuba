import json
import subprocess
import sys

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")

from avogadro_mitsuba.backend import select_variant
from avogadro_mitsuba.camera import fallback_view, from_view
from avogadro_mitsuba.render import (avogadro_light_travel,
                                     build_mitsuba_scene, run_render,
                                     srgb_to_linear, unique_path)
from avogadro_mitsuba.scene import build_scene

from conftest import molecule


def options(tmp_path, **extra):
    return {"output_dir": str(tmp_path), "width": 64, "samples": 4,
            "backend": "CPU (scalar)", **extra}


def test_scalar_variant_is_selected():
    tried = []
    assert select_variant("CPU (scalar)", tried) == "scalar_rgb"


def test_render_smoke(caffeine, tmp_path):
    result = run_render({"cjson": caffeine, "options": options(tmp_path)})
    assert "error" not in result, result
    pngs = list(tmp_path.glob("*.png"))
    assert [p.name for p in pngs] == ["molecule.png"] or len(pngs) == 1
    bitmap = np.array(mi.Bitmap(str(pngs[0])))
    assert bitmap.shape[:2] == (48, 64)  # 4:3 fallback camera
    assert (bitmap[0, 0, :3] == 255).all()  # white background
    assert bitmap[24, 32].min() < 250 or bitmap[20:28, 28:36].min() < 250


def test_transparent_background(caffeine, tmp_path):
    result = run_render({"cjson": caffeine,
                         "options": options(tmp_path, transparent=True)})
    assert "error" not in result, result
    bitmap = np.array(mi.Bitmap(str(next(tmp_path.glob("*.png")))))
    assert bitmap.shape[2] == 4
    assert bitmap[0, 0, 3] == 0
    assert bitmap[..., 3].max() == 255


def test_files_are_not_overwritten(caffeine, tmp_path):
    for _ in range(2):
        run_render({"cjson": caffeine, "options": options(tmp_path)})
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert unique_path(str(tmp_path), "x", ".png").endswith("x.png")


def test_empty_molecule_is_an_error(tmp_path):
    result = run_render({"cjson": molecule([], []), "options": options(tmp_path)})
    assert "error" in result


def test_srgb_to_linear():
    assert srgb_to_linear([0, 1]) == pytest.approx([0, 1])
    assert srgb_to_linear(0.5) == pytest.approx(0.2140, abs=1e-3)


def test_cli_entry_point(caffeine, tmp_path):
    payload = json.dumps({"cjson": caffeine, "options": options(tmp_path)})
    proc = subprocess.run(
        [sys.executable, "-c", "from avogadro_mitsuba import main; main()", "render"],
        input=payload, capture_output=True, text=True, timeout=300)
    out = [l for l in proc.stdout.splitlines() if l.startswith("{")][-1]
    assert "message" in json.loads(out), proc.stderr


def _translucent_cjson():
    cjson = molecule([6], [(0, 0, 0)])
    cjson["layer"] = {
        "visible": [True],
        "enable": {"Van der Waals": [True]},
        "settings": {"Van der Waals": ["0.25"]},
    }
    return cjson


def test_translucent_layer_is_lighter_than_opaque(tmp_path):
    opaque = _translucent_cjson()
    opaque["layer"]["settings"]["Van der Waals"] = ["1.0"]
    centre = {}
    for name, cjson in (("opaque", opaque), ("clear", _translucent_cjson())):
        out = tmp_path / name
        result = run_render({"cjson": cjson, "options": options(tmp_path, output_dir=str(out))})
        assert "error" not in result, result
        assert "opacity" not in result["message"]
        bitmap = np.array(mi.Bitmap(str(next(out.glob("*.png")))))
        centre[name] = bitmap[20:28, 28:36, :3].astype(float).mean()
    assert centre["clear"] > centre["opaque"] + 20  # more background shows


def test_translucent_scene_wraps_material_in_mask():
    scene = build_scene(_translucent_cjson())
    camera = fallback_view(scene.positions)
    mscene = build_mitsuba_scene(mi, scene, camera, 32, 24, {})
    assert str(mscene.shapes()[0].bsdf()).startswith("Mask[")
    opaque = build_scene(molecule([6], [(0, 0, 0)]))
    mscene = build_mitsuba_scene(mi, opaque, camera, 32, 24, {})
    assert not str(mscene.shapes()[0].bsdf()).startswith("Mask[")


def _view(colors, rotation=np.eye(3)):
    props = colors["properties"]
    mv = np.array(props["modelView"], dtype=float)
    mv[:3, :3] = rotation @ mv[:3, :3]
    return from_view(mv, props["projection"]), mv


def test_avogadro_light_travels_down_and_away_from_viewer(colors):
    camera, _ = _view(colors)
    travel = avogadro_light_travel(camera)
    assert travel == pytest.approx([0, -1, -1] / np.sqrt(2), abs=1e-6)


def test_avogadro_light_follows_the_camera(colors):
    angle = np.radians(70)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    camera, mv = _view(colors, rotation)
    travel = avogadro_light_travel(camera)
    # fixed in eye space: modelView maps the world direction back to -L
    eye = mv[:3, :3] @ travel
    assert eye == pytest.approx([0, -1, -1] / np.sqrt(2), abs=1e-6)
    # ... and so it differs from the unrotated world direction
    assert not np.allclose(travel, avogadro_light_travel(_view(colors)[0]))


def test_avogadro_lighting_preset_builds_a_directional_light():
    scene = build_scene(molecule([6], [(0, 0, 0)]))
    camera = fallback_view(scene.positions)
    mscene = build_mitsuba_scene(mi, scene, camera, 32, 24,
                                 {"lighting": "Avogadro-like"})
    kinds = sorted(str(e).split("[")[0] for e in mscene.emitters())
    assert kinds == ["ConstantBackgroundEmitter", "DirectionalEmitter"]


def test_translucency_option_defaults_to_match_avogadro():
    from importlib import resources
    text = (resources.files("avogadro_mitsuba") / "render-options.json").read_text()
    option = json.loads(text)["translucency"]
    assert option["type"] == "stringList"
    assert option["values"] == ["Match Avogadro", "Physical (stacked)"]
    assert option["default"] == 0


def test_composite_translucent_maths():
    from avogadro_mitsuba.render import composite_translucent
    # opaque scene: one red pixel (alpha 1) and one empty pixel
    opaque = np.array([[[0.5, 0, 0, 1.0], [0, 0, 0, 0.0]]])
    # the translucent shell drawn solid (premultiplied) over both, half
    # covering a third pixel
    solid = np.array([[[0.2, 0.4, 0.6, 1.0], [0.2, 0.4, 0.6, 1.0]],
                      [[0.1, 0.2, 0.3, 0.5], [0, 0, 0, 0]]])
    opaque = np.concatenate([opaque, np.zeros((1, 2, 4))])
    weight = np.array([[[0.25] * 4, [0.25] * 4], [[0.1] * 4, [0.0] * 4]])
    out = composite_translucent(opaque, solid, weight)
    # w * solid + (1 - w) * opaque, alpha included
    np.testing.assert_allclose(out[0, 0], [0.05 + 0.375, 0.1, 0.15, 1.0])
    np.testing.assert_allclose(out[0, 1], [0.05, 0.1, 0.15, 0.25])
    # the edge pixel is un-premultiplied: colour (0.2, 0.4, 0.6), w = 0.1
    np.testing.assert_allclose(out[1, 0], [0.02, 0.04, 0.06, 0.1], atol=1e-6)
    np.testing.assert_allclose(out[1, 1], 0.0)
    # zero weight returns the opaque render, weight is clipped to 1
    np.testing.assert_allclose(composite_translucent(opaque, solid, 0 * weight),
                               opaque)
    np.testing.assert_allclose(composite_translucent(opaque, solid, 5 + weight)[0, 0],
                               [0.2, 0.4, 0.6, 1.0])


def _translucent_over_opaque():
    """A small opaque oxygen with a large translucent sphere around it."""
    cjson = molecule([8, 6], [(0, 0, 0), (0, 0, 0)])  # red O inside grey C
    cjson["atoms"]["layer"] = [0, 1]
    cjson["layer"] = {
        "visible": [True, True],
        "enable": {"Licorice": [True, False], "Van der Waals": [False, True]},
        "settings": {"Licorice": ["1.0", ""], "Van der Waals": ["", "0.3"]},
    }
    return cjson


def test_weight_scene_has_emitters_only_on_translucent_groups():
    select_variant("CPU (scalar)")
    scene = build_scene(_translucent_over_opaque())
    camera = fallback_view(scene.positions)
    mscene = build_mitsuba_scene(mi, scene, camera, 32, 24, {}, "weight")
    assert len(mscene.shapes()) == 2
    assert len(mscene.emitters()) == 1  # no lights, one translucent group
    emitting = [s for s in mscene.shapes() if s.is_emitter()]
    assert len(emitting) == 1
    assert "0.298" in str(mscene.emitters()[0])  # 0.3 in 8 bits
    # opaque / solid scenes: no translucent bsdf, and "opaque" drops the shell
    solid = build_mitsuba_scene(mi, scene, camera, 32, 24, {}, "solid")
    assert not any(str(s.bsdf()).startswith("Mask[") for s in solid.shapes())
    opaque = build_mitsuba_scene(mi, scene, camera, 32, 24, {}, "opaque")
    assert len([x for x in opaque.shapes() if not x.is_emitter()]) == 1


def test_match_avogadro_shows_the_opaque_object_more(tmp_path):
    centre = {}
    for mode in ("Match Avogadro", "Physical (stacked)"):
        out = tmp_path / mode[:5]
        result = run_render({"cjson": _translucent_over_opaque(), "options": options(
            tmp_path, output_dir=str(out), translucency=mode, samples=16,
            width=96)})
        assert "error" not in result, result
        bitmap = np.array(mi.Bitmap(str(next(out.glob("*.png")))))
        h, w = bitmap.shape[:2]
        rgb = bitmap[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2, :3].astype(float)
        centre[mode] = (rgb[..., 0] - rgb[..., 2]).mean()  # redness
    # the red oxygen inside the grey shell: the one-layer blend lets it show
    # through more than the physical (front and back surface) shell does
    assert centre["Match Avogadro"] > centre["Physical (stacked)"] + 5
