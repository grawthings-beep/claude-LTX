import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
SIMPLE_WORKFLOW = "01_recommended_i2v_simple_10eros.json"
REFERENCE_WORKFLOW = "02_reference_ltx23_i2v_1152x896_phut_hon.json"


def load_workflow(name):
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def assert_graph_complete(testcase, workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    for link_id, origin_id, origin_slot, target_id, target_slot, _ in links.values():
        testcase.assertIn(origin_id, nodes)
        testcase.assertIn(target_id, nodes)
        testcase.assertIn(link_id, nodes[origin_id]["outputs"][origin_slot]["links"])
        testcase.assertEqual(nodes[target_id]["inputs"][target_slot]["link"], link_id)


class WorkflowTests(unittest.TestCase):
    def test_only_supported_workflows_are_bundled(self):
        names = {path.name for path in WORKFLOWS.glob("*.json")}
        self.assertEqual(names, {SIMPLE_WORKFLOW, REFERENCE_WORKFLOW})

    def test_simple_workflow_uses_10eros_stack(self):
        workflow = load_workflow(SIMPLE_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = [node["type"] for node in nodes.values()]

        self.assertEqual(
            nodes[293]["widgets_values"][0],
            "10Eros_v1-fp8mixed_learned.safetensors",
        )
        self.assertEqual(
            nodes[339]["widgets_values"][0],
            "10Eros_v1-fp8mixed_learned.safetensors",
        )
        self.assertNotIn("ImageFromBatch", types)
        self.assertNotIn("InsertImagesToBatchIndexed", types)

        lora_values = [
            node.get("widgets_values")
            for node in nodes.values()
            if node["type"] == "LoraLoaderModelOnly"
        ]
        self.assertIn(["ltx23/LTX2.3_reasoning_I2V_V3.safetensors", 1], lora_values)
        self.assertIn(
            ["ltx23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors", 0.35],
            lora_values,
        )
        assert_graph_complete(self, workflow)

    def test_reference_workflow_matches_imported_1152x896_setup(self):
        workflow = load_workflow(REFERENCE_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = [node["type"] for node in nodes.values()]

        self.assertNotIn("LTXLoopPhaseCut", types)
        self.assertIn("RIFE VFI", types)
        self.assertIn("VHS_VideoCombine", types)
        self.assertIn("ImageUpscaleWithModel", types)
        self.assertIn("UpscaleModelLoader", types)
        self.assertEqual(nodes[325]["widgets_values"], [1152, 896])
        self.assertEqual(nodes[298]["widgets_values"][0], 24)
        self.assertEqual(nodes[299]["widgets_values"][0], 162)
        self.assertEqual(
            nodes[313]["widgets_values"][0],
            "ltx-2.3-22b-dev-fp8.safetensors",
        )
        self.assertEqual(
            nodes[314]["widgets_values"],
            [
                "gemma_3_12B_it_fp4_mixed.safetensors",
                "ltx-2.3-22b-dev-fp8.safetensors",
                "default",
            ],
        )
        self.assertEqual(
            nodes[320]["widgets_values"],
            ["ltx-2.3-22b-distilled-lora-384.safetensors", 0.5],
        )

        power_loras = nodes[324]["widgets_values"]
        self.assertEqual(
            power_loras[3],
            {
                "on": True,
                "lora": "LTX23\\LTX-2.3-Phut hon.safetensors",
                "strength": 1,
                "strengthTwo": None,
            },
        )
        self.assertEqual(
            power_loras[4],
            {
                "on": True,
                "lora": "LTX23\\LTX-2-Image2Vid-Adapter.safetensors",
                "strength": 0.5,
                "strengthTwo": None,
            },
        )
        self.assertEqual(nodes[329]["widgets_values"], ["2x-AnimeSharpV4_RCAN.safetensors"])
        self.assertEqual(nodes[330]["widgets_values"], ["rife49.pth", 10, 2, False, True, 1])
        self.assertEqual(nodes[331]["widgets_values"]["frame_rate"], 32)
        self.assertIsNone(nodes[328]["inputs"][1]["link"])
        self.assertEqual(nodes[328]["mode"], 2)
        self.assertEqual(nodes[330]["mode"], 2)
        self.assertEqual(nodes[331]["mode"], 2)
        self.assertFalse(nodes[331]["widgets_values"]["save_output"])
        self.assertEqual(len(nodes[326]["outputs"]), 1)
        self.assertEqual(nodes[326]["outputs"][0]["type"], "VIDEO")
        assert_graph_complete(self, workflow)

    def test_reference_subgraph_does_not_export_frames_to_rife_by_default(self):
        workflow = load_workflow(REFERENCE_WORKFLOW)
        subgraph = workflow["definitions"]["subgraphs"][0]
        frame_outputs = [
            output
            for output in subgraph["outputs"]
            if output["name"] == "IMAGE" and output["type"] == "IMAGE"
        ]
        self.assertEqual(frame_outputs, [])

    def test_manifest_contains_supported_workflow_models(self):
        manifest = json.loads(
            (ROOT / "config" / "ltx-video-models.json").read_text(encoding="utf-8")
        )
        paths = {
            model["path"]
            for model in manifest["models"]
            if model.get("enabled", True)
        }
        expected = {
            "models/checkpoints/10Eros_v1-fp8mixed_learned.safetensors",
            "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
            "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/upscale_models/2x-AnimeSharpV4_RCAN.safetensors",
            "models/loras/ltx-2.3-22b-distilled-lora-384.safetensors",
            "models/loras/LTX23/LTX-2.3-Phut hon.safetensors",
            "models/loras/LTX23/LTX-2-Image2Vid-Adapter.safetensors",
            "models/loras/civitai/ltx23_phut_hon_civitai_2806861.safetensors",
            "models/loras/civitai/smoothmix_animations_ltx_civitai_2911845.safetensors",
            "models/loras/civitai/civitai_2849892.safetensors",
            "models/loras/ltx23/LTX2.3_reasoning_I2V_V3.safetensors",
            "models/loras/ltx23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)

    def test_custom_nodes_are_pinned(self):
        lines = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8").splitlines()
        entries = {}
        for line in lines:
            if not line or line.startswith("#"):
                continue
            name, url, revision = line.split("|")
            entries[name] = (url, revision)
            self.assertTrue(name)
            self.assertTrue(url.startswith("https://github.com/"))
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            entries["ComfyUI-Frame-Interpolation"],
            (
                "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git",
                "26545cc2dd95bc3d27f056016300673bdeee78f5",
            ),
        )

    def test_start_script_removes_retired_workflows(self):
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        self.assertIn("00_recommended_i2v_identity_lock_10eros.json", start)
        self.assertIn("02_experimental_i2v_cyclic_phasecut_10eros.json", start)
        self.assertIn("03_experimental_i2v_cyclic_phasecut_1152x896_10eros.json", start)
        self.assertIn("video_ltx23_i2v_*.json", start)
        self.assertNotIn("install_bundled_nodepacks", start)

    def test_frame_interpolation_assets_are_installed(self):
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ComfyUI-Frame-Interpolation", installer)
        self.assertIn("install.py", installer)
        self.assertIn("rife49.pth", installer)


if __name__ == "__main__":
    unittest.main()
