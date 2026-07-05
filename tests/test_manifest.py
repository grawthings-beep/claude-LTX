import json
import pathlib
import unittest

MANIFEST = pathlib.Path(__file__).resolve().parent.parent / "config" / "ltx-video-models.json"


class ManifestTests(unittest.TestCase):
    def test_manifest_is_valid_json_with_required_fields(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        models = manifest["models"]
        self.assertGreater(len(models), 0)
        for entry in models:
            self.assertIn("url", entry, entry.get("name"))
            self.assertIn("path", entry, entry.get("name"))
            self.assertTrue(entry["path"].startswith("models/"), entry["path"])

    def test_required_10eros_i2v_models_present(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        paths = {entry["path"] for entry in manifest["models"]
                 if entry.get("enabled", True)}
        expected = {
            "models/checkpoints/10Eros_v1-fp8mixed_learned.safetensors",
            "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/loras/ltx23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
            "models/loras/ltx23/LTX2.3_reasoning_I2V_V3.safetensors",
            "models/loras/ltx23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)


if __name__ == "__main__":
    unittest.main()
