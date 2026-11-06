import json
import subprocess
import sys

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")

from avogadro_mitsuba.backend import select_variant
from avogadro_mitsuba.render import run_render, srgb_to_linear, unique_path

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
