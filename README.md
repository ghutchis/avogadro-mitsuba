# avogadro-mitsuba

Path-traced molecular renderings from Avogadro 2 using
[Mitsuba 3](https://www.mitsuba-renderer.org/).

Adds **File → Export → Path-Traced Image (Mitsuba)...**. The molecule is sent
as CJSON, so the saved view, custom atom colors, bond orders and the layer
visibility / Ball and Stick, Licorice and Van der Waals settings carry over.

Options: output directory, image width, samples per pixel, geometry detail,
material (glossy, matte, metallic), transparent background, backend
(auto: CUDA, Metal, LLVM, then scalar CPU).

Not yet handled: other styles (Wireframe, Cartoons), surfaces and orbitals,
layer opacity below 1 (ignored, with a note).

Regenerate the element table with `python scripts/gen_elements.py`; run the
tests with `pixi run -e test pytest`.
