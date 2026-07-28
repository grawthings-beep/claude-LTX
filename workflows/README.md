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

`loop.json` keeps the intact two-stage I2V path and generates 193 unconstrained
frames. It does not add a final-frame guide or generate a return bridge.
`Cyclic Loop Phase Cut` compares the appearance and temporal velocity of each
candidate 152-frame window, selects the best phase, and overlap-adds only its
first and last 8 frames. The output is 144 frames at 24 fps, exactly 6 seconds.
Audio is cut and overlapped at the same phase.
