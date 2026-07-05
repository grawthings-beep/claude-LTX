# Workflows

Recommended first:

- `00_recommended_i2v_identity_lock_10eros.json`
- `01_recommended_i2v_simple_10eros.json`

Both use the 10Eros checkpoint, Reasoning I2V, and a light Edit Anything IC LoRA
pass. They do not need `CIVITAI_TOKEN`.

Other useful workflows:

- `video_ltx23_i2v_first_last_pair_dasiwa_fast.json`: separate first/last images
- `video_ltx23_i2v_first_last_same_dasiwa_fast.json`: loop/identity lock
- `video_ltx23_i2v_simple_dasiwa_fast.json`: faster single-image I2V
- `video_ltx23_i2v_*_2stage_hq.json`: higher quality, Civitai LoRA dependent

If a workflow complains about a missing `civitai/...` LoRA, either set
`CIVITAI_TOKEN` and let the downloader fetch optional Civitai models, or switch
to one of the `00_`, `01_`, or `*_dasiwa_*` workflows.
