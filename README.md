# avogadro-mitsuba

**Prototype.** Path-traced molecular renderings from Avogadro 2 using
[Mitsuba 3](https://www.mitsuba-renderer.org/). Expect rough edges.

Adds **File → Export → Path-Traced Image (Mitsuba)...**. The molecule is sent
as CJSON, so the saved view, custom atom colors, bond orders and the layer
visibility / Ball and Stick, Licorice and Van der Waals settings (including
layer opacity) carry over.

Options: output directory, image width, samples per pixel, geometry detail,
material (glossy, matte, metallic), lighting (Studio, or Avogadro-like for a
flat light from above and in front as in the Avogadro window), transparent
background, backend.

## Requirements

- Avogadro 2.1 or newer.
- Mitsuba 3.9 or newer (installed from PyPI; only 3.9.1 has been tested).

## Platforms

- macOS on Apple Silicon: renders on the GPU with Metal.
- Linux and Windows with an NVIDIA GPU: CUDA.
- Otherwise the scalar CPU backend, which is much slower; lower the samples
  per pixel or the image width.
- Intel Macs are not supported (Mitsuba has no macOS x86_64 wheel).
- The LLVM backend needs libLLVM and is not wired up yet, so Auto skips it in
  practice.

## Limitations

- Only Ball and Stick, Licorice and Van der Waals layers are drawn; Wireframe
  is skipped, and there are no cartoons, surfaces, orbitals or labels.
- Avogadro-like lighting is an approximation: the light is fixed to the view,
  but the path tracer still adds soft shadows and reflections.

Regenerate the element table with `python scripts/gen_elements.py`; run the
tests with `pixi run -e test pytest`.
