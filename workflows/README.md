# Workflow

Bundled workflows:

- `mrxin-i2v.json`
- `mrxin-i2v-hq.json`
- `mrxin-i2v-auto-mosaic.json`: first pass with CPU mosaic, no upscale.
- `mrxin-i2v-2stage-auto-mosaic.json`: both LTX stages with CPU mosaic.
- `mrxin-i2v-nmkd-auto-mosaic.json`: first pass, NMKD-Siax 2x, CPU mosaic.

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

The experimental `mrxin-i2v-hq.json` keeps the same models, LoRAs, samplers,
audio path, and five-second duration. Its first pass is 896x1184 and the latent
x2 pass produces 1792x2368. The source image is resized once to that final
resolution and sent through `LTXVPreprocess` to both I2V stages without the
original 1536-pixel longer-edge reduction.

The two-stage auto-mosaic version keeps both LTX sampling stages, with a
512x704 first pass and 1024x1408 latent x2 final pass. Models, LoRAs, sampling,
frame count, FPS and audio settings are unchanged. Mosaic follows the final decode.

The NMKD version generates its first pass at 512x704 but does not run a final
LTX sampling pass. Decoded frames go through `4x_NMKD-Siax_200k.pth` and a
nearest-exact 0.5x resize for a 1024x1408 result. Mosaic runs once after resizing,
immediately before the only MP4 encoder. Audio and FPS stay on the first-pass
path. Both upscale mosaic versions' size controls specify the final output size;
generation is half that width and height. Existing LoRA settings and input-image
conditioning are kept.

The manifest downloads every model selected by the workflow's default active
path. The custom-node list pins every external node pack used by the graph.
