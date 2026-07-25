# ComfyUI-LTXLoop

Experimental seamless-loop nodes for LTX 2.3:

- `LTX Mobius Sampler` applies training-free cyclic latent shifting during
  denoising. Packed LTX video and audio latents are shifted together.
- `LTX Loop Decode` gives the causal video VAE wrapped tail/head context and
  crops the central cycle.
- `LTX Loop Audio Seam` smooths the audio boundary without changing duration.

The sampler follows the latent-shift direction introduced by
[Mobius](https://github.com/YisuiTT/Mobius). It is an LTX-specific experimental
port, not an official Lightricks node.

The bundled `loop.json` does not use the experimental Mobius sampler. It uses
ComfyUI's official `LTXVAddGuide` first/last-frame conditioning and only uses
`LTX Loop Audio Seam` from this pack.
