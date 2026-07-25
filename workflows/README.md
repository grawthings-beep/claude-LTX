# Workflows

Bundled workflows:

- `original.json`
- `i2v.json`
- `loop.json`

`original.json` preserves the uploaded reference graph, including its AnimeSharp
and RIFE side-output branch. Its requested model and output settings are 10Eros
and `1728x1152`.

`i2v.json` keeps the same graph and model stack but disables the optional RIFE
side-output branch. The main LTX output keeps its original audio timing.

`loop.json` starts from the same intact graph as `i2v.json`. In both the low-
and high-resolution sampling stages, the input image is passed to the built-in
`LTXVAddGuide` node at frame `0` and frame `-1`, each at strength `0.7`. This is
the FLF2V conditioning pattern used by the official ComfyUI LTX workflow, not a
final-frame image overwrite. The audio boundary is smoothed without changing
duration. The workflow uses `161` frames to follow LTX's `8n+1` layout.
