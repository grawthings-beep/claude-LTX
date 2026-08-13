import json
import pathlib
import unittest


MANIFEST = pathlib.Path(__file__).resolve().parent.parent / "config" / "ltx-video-models.json"


def load_models():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]


class ManifestTests(unittest.TestCase):
    def test_manifest_is_valid_json_with_required_fields(self):
        models = load_models()
        self.assertGreater(len(models), 0)
        for entry in models:
            self.assertIn("url", entry, entry.get("name"))
            self.assertIn("path", entry, entry.get("name"))
            self.assertTrue(entry["path"].startswith("models/"), entry["path"])

    def test_mrxin_workflow_models_are_present(self):
        paths = {
            entry["path"] for entry in load_models() if entry.get("enabled", True)
        }
        expected = {
            "models/checkpoints/ltx2310eros_beta.safetensors",
            "models/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
            "models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/vae/LTX23_video_vae_bf16.safetensors",
            "models/vae/LTX23_audio_vae_bf16.safetensors",
            "models/vae/taeltx2_3.safetensors",
            "models/upscale_models/nmkdSiaxCX_200k.safetensors",
            "models/loras/LTX2/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
            "models/loras/LTX 2.3/LTX2.3_Reasoning_V1.safetensors",
            "models/loras/LTX2/DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors",
            "models/loras/LTX2/LTX2_3_NSFW_furry_concat_v2.safetensors",
            "models/loras/LTX 2.3/LTX-2.3 - Orgasm.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)

    def test_blowjob_lora_is_the_first_download(self):
        models = load_models()
        entry = models[0]

        self.assertEqual(entry["priority"], -100)
        self.assertEqual(
            entry["url"],
            "https://civitai.red/api/download/models/2849892?fileId=2736052&token=${CIVITAI_TOKEN}",
        )
        self.assertEqual(
            entry["path"],
            "models/loras/ltx23/LTX2.3_blowjob_animation_I2V_v1.0.safetensors",
        )
        self.assertEqual(
            entry["sha256"],
            "3aaf3a7b3384637694277fa9a211b0d4956da7f5512509dc420282ca033c8a6e",
        )

    def test_existing_lora_downloads_are_preserved(self):
        paths = {entry["path"] for entry in load_models()}
        expected = {
            "models/loras/ltx23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
            "models/loras/ltx-2.3-22b-distilled-lora-384.safetensors",
            "models/loras/LTX23/LTX-2.3-Phut hon.safetensors",
            "models/loras/LTX23/LTX-2-Image2Vid-Adapter.safetensors",
            "models/loras/civitai/ltx23_phut_hon_civitai_2806861.safetensors",
            "models/loras/civitai/smoothmix_animations_ltx_civitai_2911845.safetensors",
            "models/loras/ltx23/LTX2.3_reasoning_I2V_V3.safetensors",
            "models/loras/ltx23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",
            "models/loras/ltx23/LTX-2.3jiggle.safetensors",
            "models/loras/ltx23/LTX2.3_blowjob_animation_I2V_v1.0.safetensors",
            "models/loras/ltx23/throat_bulge-10Eros_i2v_v1.0.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)

    def test_default_workflow_models_have_startup_priority(self):
        by_path = {entry["path"]: entry for entry in load_models()}
        startup_paths = {
            "models/checkpoints/ltx2310eros_beta.safetensors",
            "models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/vae/LTX23_video_vae_bf16.safetensors",
            "models/vae/LTX23_audio_vae_bf16.safetensors",
            "models/vae/taeltx2_3.safetensors",
            "models/loras/LTX 2.3/LTX2.3_Reasoning_V1.safetensors",
            "models/loras/LTX2/DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors",
            "models/loras/LTX2/LTX2_3_NSFW_furry_concat_v2.safetensors",
            "models/loras/LTX 2.3/LTX-2.3 - Orgasm.safetensors",
        }
        self.assertTrue(all(by_path[path]["priority"] == 0 for path in startup_paths))


if __name__ == "__main__":
    unittest.main()
