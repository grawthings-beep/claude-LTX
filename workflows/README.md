# Workflow

Bundled workflow:

- `mrxin-i2v.json`

This is the unmodified `MrXin LTX 2.3 I2V EROS V4` workflow JSON from Civitai
model version `2835183` (`mrxinLTX23I2VEros12GBVRAM_i2vV40.zip`). It includes the 10Eros checkpoint and
distilled-model switch, two-stage LTX I2V generation, audio, latent upscaling,
an optional video editor, RIFE interpolation, and optional RTX video super
resolution.

The author's defaults select the 10Eros model and standalone CLIP, video VAE,
and audio VAE branches. The original active content LoRA stack and dynamic
rank-105 distilled LoRA are preserved.

The manifest downloads every model selected by the workflow's default active
path. The custom-node list pins every external node pack used by the graph.
