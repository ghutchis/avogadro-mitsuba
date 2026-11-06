import copy
import json
import os
from pathlib import Path

import pytest

DATA_DIR = Path(os.environ.get(
    "AVOGADRO_DATA_ROOT", Path.home() / "Devel/openchemistry/avogadrodata/data"
)) / "cjson"


def load_cjson(name):
    path = DATA_DIR / f"{name}.cjson"
    if not path.exists():
        pytest.skip(f"{path} not found (set AVOGADRO_DATA_ROOT)")
    return json.loads(path.read_text())


@pytest.fixture
def caffeine():
    return load_cjson("caffeine")


@pytest.fixture
def colors():
    return load_cjson("colors")


@pytest.fixture
def aniline():
    return load_cjson("aniline")


def molecule(numbers, coords, bonds=(), orders=()):
    """A minimal CJSON with no layer or camera block."""
    return {
        "atoms": {"elements": {"number": list(numbers)},
                  "coords": {"3d": [c for xyz in coords for c in xyz]}},
        "bonds": {"connections": {"index": [i for b in bonds for i in b]},
                  "order": list(orders)},
    }


@pytest.fixture
def clone():
    return copy.deepcopy
