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

    def test_required_official_models_present(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        paths = {entry["path"] for entry in manifest["models"]
                 if entry.get("enabled", True)}
        expected = {
            "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
            "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/loras/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)


if __name__ == "__main__":
    unittest.main()
