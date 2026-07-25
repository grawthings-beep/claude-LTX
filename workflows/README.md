# Workflows

Bundled workflows:

- `i2v.json`
- `loop.json`

`i2v.json` is the normal LTX 2.3 I2V workflow. It uses the 10Eros checkpoint,
the official distilled 384 LoRA at `0.5`, the LTX 2.3 dev fp8 audio stack, and
the LTX spatial x2 upscaler. Its final size is `1728x1152`.

`loop.json` keeps the same model and LoRA stack but replaces both samplers with
the bundled `LTX Mobius Sampler`. The sampler cyclically shifts video and audio
latents during denoising. `LTX Loop Decode` wraps temporal context around the
causal video VAE, and `LTX Loop Audio Seam` smooths the audio boundary. The loop
workflow uses `161` frames so its latent/video duration follows LTX's `8n+1`
temporal layout.
