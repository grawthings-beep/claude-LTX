import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
I2V_WORKFLOW = "i2v.json"
LOOP_WORKFLOW = "loop.json"
ORIGINAL_WORKFLOW = "original.json"


def load_workflow(name):
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def assert_root_graph_complete(testcase, workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    for link_id, origin_id, origin_slot, target_id, target_slot, _ in links.values():
        testcase.assertIn(origin_id, nodes)
        testcase.assertIn(target_id, nodes)
        testcase.assertIn(link_id, nodes[origin_id]["outputs"][origin_slot]["links"])
        testcase.assertEqual(nodes[target_id]["inputs"][target_slot]["link"], link_id)


def assert_subgraph_complete(testcase, subgraph):
    nodes = {node["id"]: node for node in subgraph["nodes"]}
    links = {link["id"]: link for link in subgraph["links"]}
    for link in links.values():
        origin_id = link["origin_id"]
        target_id = link["target_id"]
        if origin_id != -10:
            testcase.assertIn(origin_id, nodes)
            testcase.assertIn(
                link["id"],
                nodes[origin_id]["outputs"][link["origin_slot"]]["links"],
            )
        if target_id != -20:
            testcase.assertIn(target_id, nodes)
            testcase.assertEqual(
                nodes[target_id]["inputs"][link["target_slot"]]["link"],
                link["id"],
            )


class WorkflowTests(unittest.TestCase):
    def test_only_simple_workflow_names_are_bundled(self):
        names = {path.name for path in WORKFLOWS.glob("*.json")}
        self.assertEqual(names, {ORIGINAL_WORKFLOW, I2V_WORKFLOW, LOOP_WORKFLOW})

    def test_original_workflow_keeps_uploaded_graph(self):
        workflow = load_workflow(ORIGINAL_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}

        self.assertEqual(len(nodes), 25)
        self.assertEqual(nodes[325]["type"], "SetImageSize")
        self.assertEqual(nodes[325]["widgets_values"], [1728, 1152])
        self.assertEqual(nodes[328]["type"], "ImageUpscaleWithModel")
        self.assertEqual(nodes[328]["mode"], 0)
        self.assertEqual(nodes[329]["type"], "UpscaleModelLoader")
        self.assertEqual(nodes[330]["type"], "RIFE VFI")
        self.assertEqual(nodes[330]["mode"], 0)
        self.assertEqual(nodes[331]["type"], "VHS_VideoCombine")
        self.assertEqual(nodes[331]["mode"], 0)
        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, workflow["definitions"]["subgraphs"][0])

    def test_i2v_workflow_matches_requested_stack(self):
        workflow = load_workflow(I2V_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = [node["type"] for node in nodes.values()]

        self.assertEqual(
            nodes[313]["widgets_values"],
            ["10Eros_v1-fp8mixed_learned.safetensors"],
        )
        self.assertEqual(
            nodes[280]["widgets_values"],
            ["10Eros_v1-fp8mixed_learned.safetensors"],
        )
        self.assertEqual(
            nodes[314]["widgets_values"],
            [
                "gemma_3_12B_it_fp4_mixed.safetensors",
                "10Eros_v1-fp8mixed_learned.safetensors",
                "default",
            ],
        )
        self.assertEqual(
            nodes[320]["widgets_values"],
            ["ltx-2.3-22b-distilled-lora-384.safetensors", 0.5],
        )
        self.assertEqual(
            nodes[309]["widgets_values"],
            ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"],
        )
        self.assertEqual(nodes[325]["widgets_values"], [1728, 1152])
        self.assertEqual(nodes[325]["type"], "SetImageSize")
        self.assertEqual(nodes[328]["type"], "ImageUpscaleWithModel")
        self.assertEqual(nodes[328]["mode"], 2)
        self.assertEqual(nodes[330]["type"], "RIFE VFI")
        self.assertEqual(nodes[330]["mode"], 2)
        self.assertEqual(nodes[331]["type"], "VHS_VideoCombine")
        self.assertEqual(nodes[331]["mode"], 2)
        self.assertEqual(nodes[298]["widgets_values"][0], 24)
        self.assertEqual(nodes[299]["widgets_values"][0], 162)
        self.assertEqual(nodes[75]["widgets_values"][0], "video/i2v")
        self.assertNotIn("LTXMobiusSampler", types)

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
        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, workflow["definitions"]["subgraphs"][0])

    def test_loop_workflow_guides_both_ends_in_both_sampling_stages(self):
        workflow = load_workflow(LOOP_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        subgraph = workflow["definitions"]["subgraphs"][0]
        subnodes = {node["id"]: node for node in subgraph["nodes"]}
        sublinks = {link["id"]: link for link in subgraph["links"]}

        self.assertEqual(
            nodes[313]["widgets_values"],
            ["10Eros_v1-fp8mixed_learned.safetensors"],
        )
        self.assertEqual(
            nodes[280]["widgets_values"],
            ["10Eros_v1-fp8mixed_learned.safetensors"],
        )
        self.assertEqual(
            nodes[314]["widgets_values"],
            [
                "gemma_3_12B_it_fp4_mixed.safetensors",
                "10Eros_v1-fp8mixed_learned.safetensors",
                "default",
            ],
        )
        self.assertEqual(
            nodes[320]["widgets_values"],
            ["ltx-2.3-22b-distilled-lora-384.safetensors", 0.5],
        )
        self.assertEqual(nodes[325]["widgets_values"], [1728, 1152])
        self.assertEqual(nodes[325]["type"], "SetImageSize")
        self.assertEqual(nodes[328]["type"], "ImageUpscaleWithModel")
        self.assertEqual(nodes[328]["mode"], 2)
        self.assertEqual(nodes[330]["type"], "RIFE VFI")
        self.assertEqual(nodes[330]["mode"], 2)
        self.assertEqual(nodes[331]["type"], "VHS_VideoCombine")
        self.assertEqual(nodes[331]["mode"], 2)
        self.assertEqual(nodes[299]["widgets_values"][0], 161)
        self.assertEqual(nodes[75]["widgets_values"][0], "video/loop")

        self.assertEqual(subnodes[289]["type"], "KSamplerSelect")
        self.assertEqual(subnodes[281]["type"], "KSamplerSelect")
        self.assertEqual(subnodes[312]["type"], "VAEDecodeTiled")
        self.assertEqual(
            {
                node_id: (subnodes[node_id]["type"], subnodes[node_id]["widgets_values"])
                for node_id in (294, 333, 286, 334)
            },
            {
                294: ("LTXVAddGuide", [0, 0.7]),
                333: ("LTXVAddGuide", [-1, 0.7]),
                286: ("LTXVAddGuide", [0, 0.7]),
                334: ("LTXVAddGuide", [-1, 0.7]),
            },
        )
        self.assertEqual(subnodes[335]["type"], "LTXLoopAudioSeam")
        self.assertEqual(subnodes[335]["widgets_values"], [120])
        self.assertEqual(
            sublinks[743],
            {
                "id": 743,
                "origin_id": 295,
                "origin_slot": 0,
                "target_id": 335,
                "target_slot": 0,
                "type": "AUDIO",
            },
        )
        self.assertEqual(subnodes[308]["inputs"][1]["link"], 686)

        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, subgraph)

    def test_i2v_and_loop_keep_original_root_topology(self):
        original = load_workflow(ORIGINAL_WORKFLOW)
        expected_nodes = [(node["id"], node["type"]) for node in original["nodes"]]
        expected_links = original["links"]

        for name in (I2V_WORKFLOW, LOOP_WORKFLOW):
            workflow = load_workflow(name)
            self.assertEqual(
                [(node["id"], node["type"]) for node in workflow["nodes"]],
                expected_nodes,
            )
            self.assertEqual(workflow["links"], expected_links)

    def test_optional_loras_are_available_in_all_workflows(self):
        expected = {
            "ltx23\\LTX-2.3jiggle.safetensors",
            "ltx23\\LTX2.3_blowjob_animation_I2V_v1.0.safetensors",
            "ltx23\\throat_bulge-10Eros_i2v_v1.0.safetensors",
        }
        for name in (ORIGINAL_WORKFLOW, I2V_WORKFLOW, LOOP_WORKFLOW):
            workflow = load_workflow(name)
            nodes = {node["id"]: node for node in workflow["nodes"]}
            power_loras = nodes[324]["widgets_values"]
            optional_loras = {
                widget["lora"]
                for widget in power_loras
                if isinstance(widget, dict)
                and "lora" in widget
                and not widget["on"]
            }
            self.assertTrue(expected.issubset(optional_loras), expected - optional_loras)

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
            "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "models/upscale_models/2x-AnimeSharpV4_RCAN.safetensors",
            "models/loras/ltx-2.3-22b-distilled-lora-384.safetensors",
            "models/loras/LTX23/LTX-2.3-Phut hon.safetensors",
            "models/loras/LTX23/LTX-2-Image2Vid-Adapter.safetensors",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)

    def test_custom_nodes_are_pinned(self):
        lines = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8").splitlines()
        names = set()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            name, url, revision = line.split("|")
            names.add(name)
            self.assertTrue(name)
            self.assertTrue(url.startswith("https://github.com/"))
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            names,
            {
                "10S_Nodes",
                "rgthree-comfy",
                "ComfyUI-VideoHelperSuite",
                "ComfyUI-Frame-Interpolation",
            },
        )

    def test_bundled_loop_nodes_are_installed(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        node_source = (
            ROOT / "custom_nodepacks" / "ComfyUI-LTXLoop" / "nodes.py"
        ).read_text(encoding="utf-8")

        self.assertIn("COPY custom_nodepacks/", dockerfile)
        self.assertIn("install_bundled_nodepacks", installer)
        self.assertIn("rife49.pth", installer)
        self.assertIn('"SetImageSize": LTXSetImageSize', node_source)
        self.assertIn('"LTXSetImageSize": LTXSetImageSize', node_source)
        self.assertIn('"LTXMobiusSampler": LTXMobiusSampler', node_source)
        self.assertIn('"LTXLoopDecodeTiled": LTXLoopDecodeTiled', node_source)
        self.assertIn('"LTXLoopAudioSeam": LTXLoopAudioSeam', node_source)

    def test_image_is_slimmed_and_waits_for_cuda(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

        self.assertIn("runpod/comfyui:1.4.1-cuda12.8@sha256:", dockerfile)
        self.assertIn(
            "find /opt/comfyui-baked/custom_nodes -mindepth 1 -maxdepth 1",
            dockerfile,
        )
        self.assertIn("check_workflow_nodes.py", dockerfile)
        self.assertIn("wait_for_gpu", start)
        self.assertIn("GPU_WAIT_TIMEOUT:-600", start)
        self.assertIn('ctypes.CDLL("libcuda.so.1")', start)
        self.assertIn("nvidia-smi --query-gpu=name", start)
        self.assertLess(start.index("start_background_downloads"), start.rindex("wait_for_gpu"))

    def test_start_script_preserves_existing_workflows(self):
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        self.assertNotIn('rm -f "${COMFYUI_WORKFLOW_DIR}"', start)


if __name__ == "__main__":
    unittest.main()
