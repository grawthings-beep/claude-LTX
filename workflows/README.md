# Workflow

Bundled workflow:

- `mrxin-i2v.json`

This is a RunPod-adapted `MrXin LTX 2.3 I2V EROS V4` workflow imported from
`mrxinLTX23I2VEros12GBVRAM_i2vV40.zip`. It includes the 10Eros checkpoint and
distilled-model switch, two-stage LTX I2V generation, audio, latent upscaling,
an optional video editor, RIFE interpolation, and optional RTX video super
resolution.

The default switch group consistently selects the full 10Eros checkpoint for
the model, CLIP projection, video VAE, and audio VAE. Optional content LoRAs are
kept in the loader but disabled for a clean baseline; the active distilled LoRA
is the I2V-safe TenStrip `condsafe` variant.

The manifest downloads every model selected by the workflow's default active
path. Models used only by disabled LoRA slots remain user-selectable when they
are already present or added later.
