# ComfyUI-LTXLoop

Experimental seamless-loop nodes for LTX 2.3:

- `LTX Loop Bridge Frames` extracts tail/head `8n+1` motion clips from a
  generated video for a second LTX conditioning pass.
- `LTX Loop Assemble` removes duplicate guide contexts, joins the base and
  bridge frame batches, and smooths both generated-audio boundaries.
- `LTX Mobius Sampler` applies training-free cyclic latent shifting during
  denoising. Packed LTX video and audio latents are shifted together.
- `LTX Loop Decode` gives the causal video VAE wrapped tail/head context and
  crops the central cycle.
- `LTX Loop Audio Seam` smooths the audio boundary without changing duration.

The sampler follows the latent-shift direction introduced by
[Mobius](https://github.com/YisuiTT/Mobius). It is an LTX-specific experimental
port, not an official Lightricks node.

The bundled `loop.json` does not use the experimental Mobius sampler. It keeps
the original two-stage I2V generation as pass 1, then uses ComfyUI's
`LTXVAddGuide` with 9-frame tail/head motion contexts in a separate two-stage
bridge pass. The bridge guides are cropped before decode, and `LTX Loop
Assemble` removes the duplicate contexts.

The pack also registers `SetImageSize` as a compatibility alias so the original
workflow keeps its node type unchanged.
