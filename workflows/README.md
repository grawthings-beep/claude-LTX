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

`loop.json` uses a two-pass bridge design. Pass 1 is the intact two-stage I2V
path and generates 129 frames without a forced final still. Pass 2 extracts the
tail and head 9-frame motion contexts, then generates a 49-frame LTX bridge in
two stages. `LTXVCropGuides` runs after both bridge sampling stages so guide
latents never appear as output frames. Removing the repeated contexts produces
160 frames at 24 fps. The generated audio is joined with boundary smoothing.
