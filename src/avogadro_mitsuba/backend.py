"""Pick a Mitsuba variant: the fastest one that works on this machine."""

import sys

# "Backend" option -> variants to try in order
CHOICES = {
    "Auto": ["cuda_ad_rgb", "metal_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
    "CUDA": ["cuda_ad_rgb"],
    "Metal": ["metal_ad_rgb"],
    "LLVM": ["llvm_ad_rgb"],
    "CPU (scalar)": ["scalar_rgb"],
}


def select_variant(choice="Auto", tried=None):
    """Set the Mitsuba variant and return its name.

    Each candidate is tried in turn; failures (missing GPU, missing libLLVM,
    variant not in this wheel) are recorded in `tried` if given. Always falls
    back to scalar_rgb unless a specific backend was requested.
    """
    import mitsuba as mi

    candidates = list(CHOICES.get(choice, CHOICES["Auto"]))
    if sys.platform != "darwin":
        candidates = [c for c in candidates if not c.startswith("metal")] or candidates
    if choice != "Auto" and "scalar_rgb" not in candidates:
        candidates.append("scalar_rgb")

    available = set(mi.variants())
    for name in candidates:
        if name not in available:
            if tried is not None:
                tried.append(f"{name}: not in this Mitsuba build")
            continue
        try:
            mi.set_variant(name)
            if name != "scalar_rgb":
                # JIT variants may only fail on first use (no GPU / libLLVM)
                mi.Float(1.0)
            return name
        except Exception as exc:
            if tried is not None:
                tried.append(f"{name}: {str(exc).splitlines()[0] if str(exc) else exc}")
    raise RuntimeError("No usable Mitsuba variant found: " + "; ".join(tried or []))
