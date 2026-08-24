# Workflow

Bundled workflow:

- `mrxin-i2v.json`

This keeps the `MrXin LTX 2.3 I2V EROS V4` graph from Civitai model version
`2835183` (`mrxinLTX23I2VEros12GBVRAM_i2vV40.zip`). It includes the 10Eros checkpoint and
distilled-model switch, two-stage LTX I2V generation, audio, latent upscaling,
an optional video editor, RIFE interpolation, and optional RTX video super
resolution.

The default path consistently selects the complete 10Eros checkpoint for the
model, CLIP, video VAE, and audio VAE. It uses the `condsafe` distilled LoRA at
0.6, disables content LoRAs, applies 0.7 first-pass image conditioning and 1.0
final-pass conditioning, enables the last-frame VAE fix, and generates five
seconds at 960x1280. Content LoRAs are preserved for deliberate one-at-a-time
experiments.

The manifest downloads every model selected by the workflow's default active
path. The custom-node list pins every external node pack used by the graph.
