# Workflows

Bundled workflows:

- `01_recommended_i2v_simple_10eros.json`
- `02_reference_ltx23_i2v_1152x896_phut_hon.json`

`01_recommended_i2v_simple_10eros.json` is the simple 10Eros I2V workflow.

`02_reference_ltx23_i2v_1152x896_phut_hon.json` is the imported reference
workflow from `a914e520-9bc7-49bb-835f-b6717cd19ba3.json`. It preserves the
`1152x896` LTX 2.3 I2V setup, the Phut hon and Image2Vid Adapter LoRAs, and the
RIFE/VHS side branch from the source workflow. The side branch is disabled and
does not save output by default because its `10x` interpolation settings change
video duration unless the output frame rate and audio path are rebuilt together.
The RIFE node comes from the `ComfyUI-Frame-Interpolation` custom node pack
pinned in `custom_nodes.txt`. The image build also preinstalls `rife49.pth`; the
AnimeSharpV4 x2 RCAN upscaler is listed in the model manifest.
