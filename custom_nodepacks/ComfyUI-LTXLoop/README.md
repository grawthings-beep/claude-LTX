# ComfyUI-LTXLoop

Experimental seamless-loop nodes for LTX 2.3:

- `Cyclic Loop Phase Cut` finds the generated window whose head/tail appearance
  and temporal velocity match best, then overlap-adds video and audio at that
  phase. It never asks LTX to regenerate a return to the input image.
- `LTX Mobius Sampler` applies training-free cyclic latent shifting during
  denoising. Packed LTX video and audio latents are shifted together.
- `LTX Loop Decode` gives the causal video VAE wrapped tail/head context and
  crops the central cycle.
- `LTX Loop Audio Seam` smooths the audio boundary without changing duration.

The sampler follows the latent-shift direction introduced by
[Mobius](https://github.com/YisuiTT/Mobius). It is an LTX-specific experimental
port, not an official Lightricks node.

The bundled `loop.json` does not use the experimental Mobius sampler. It keeps
the original two-stage I2V generation, creates extra unconstrained motion for
phase selection, then uses `Cyclic Loop Phase Cut` to produce a 6-second cycle.

The pack also registers `SetImageSize` as a compatibility alias so the original
workflow keeps its node type unchanged.
