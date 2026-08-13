import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
I2V_WORKFLOW = "mrxin-i2v.json"
WORKFLOW_SHA256 = "80445825a2dfba41a02a0973ea3f5cf1cd1fb4b35114b2c9ee2701b8b0de8183"


def load_workflow():
    return json.loads((WORKFLOWS / I2V_WORKFLOW).read_text(encoding="utf-8"))


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
    def test_only_mrxin_i2v_workflow_is_bundled(self):
        names = {path.name for path in WORKFLOWS.glob("*.json")}
        self.assertEqual(names, {I2V_WORKFLOW})

    def test_runpod_adapted_workflow_is_locked(self):
        digest = hashlib.sha256((WORKFLOWS / I2V_WORKFLOW).read_bytes()).hexdigest()
        self.assertEqual(digest, WORKFLOW_SHA256)

    def test_workflow_graph_is_complete(self):
        workflow = load_workflow()
        self.assertEqual(len(workflow["nodes"]), 97)
        self.assertEqual(len(workflow["definitions"]["subgraphs"]), 1)
        self.assertEqual(len(workflow["definitions"]["subgraphs"][0]["nodes"]), 47)
        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, workflow["definitions"]["subgraphs"][0])

    def test_workflow_model_stack_and_defaults_are_preserved(self):
        nodes = {node["id"]: node for node in load_workflow()["nodes"]}

        self.assertEqual(nodes[1]["widgets_values"], ["ltx2310eros_beta.safetensors"])
        self.assertEqual(
            nodes[186]["widgets_values"],
            [
                "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
                "default",
            ],
        )
        self.assertEqual(
            nodes[118]["widgets_values"],
            [
                "gemma_3_12B_it_fp8_e4m3fn.safetensors",
                "ltx2310eros_beta.safetensors",
                "default",
            ],
        )
        self.assertEqual(
            nodes[189]["widgets_values"],
            [
                "gemma_3_12B_it_fp8_e4m3fn.safetensors",
                "ltx-2.3_text_projection_bf16.safetensors",
                "ltxv",
                "default",
            ],
        )
        self.assertEqual(nodes[188]["widgets_values"], ["LTX23_video_vae_bf16.safetensors"])
        self.assertEqual(nodes[190]["widgets_values"], ["LTX23_audio_vae_bf16.safetensors"])
        self.assertEqual(nodes[5]["widgets_values"], ["taeltx2_3.safetensors"])
        self.assertEqual(
            nodes[4]["widgets_values"],
            ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"],
        )
        self.assertEqual(nodes[170]["widgets_values"], ["nmkdSiaxCX_200k.safetensors"])
        self.assertEqual(nodes[18]["widgets_values"], [20, 20, 0])
        self.assertEqual(nodes[19]["widgets_values"], [704, 704, 0])
        self.assertEqual(nodes[181]["widgets_values"], [1280, 1280, 0])
        self.assertTrue(
            all(nodes[node_id]["widgets_values"] == [1] for node_id in (191, 192, 193, 194))
        )

    def test_default_lora_stack_is_stable(self):
        nodes = {node["id"]: node for node in load_workflow()["nodes"]}
        active = {
            widget["lora"]
            for widget in nodes[6]["widgets_values"]
            if isinstance(widget, dict) and widget.get("on") and widget.get("lora")
        }
        self.assertEqual(active, set())
        configured = {
            widget["lora"]
            for widget in nodes[6]["widgets_values"]
            if isinstance(widget, dict) and widget.get("lora")
        }
        self.assertIn("LTX 2.3\\LTX2.3_Reasoning_V1.safetensors", configured)
        self.assertIn("LTX2\\DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors", configured)
        self.assertIn("LTX2\\LTX2_3_NSFW_furry_concat_v2.safetensors", configured)
        self.assertIn("LTX 2.3\\LTX-2.3 - Orgasm.safetensors", configured)
        distilled = nodes[7]["widgets_values"][2]
        self.assertTrue(distilled["on"])
        self.assertEqual(
            distilled["lora"],
            "ltx23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        )
        self.assertEqual(distilled["strength"], 0.6)

    def test_required_custom_nodes_are_pinned(self):
        lines = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8").splitlines()
        names = set()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            name, url, revision = line.split("|")
            names.add(name)
            self.assertTrue(url.startswith("https://github.com/"))
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

        required = {
            "rgthree-comfy",
            "ComfyUI-VideoHelperSuite",
            "ComfyUI-KJNodes",
            "ComfyUI-mxToolkit",
            "ComfyUI-Easy-Use",
            "ComfyUI-Impact-Pack",
            "ComfyUI-Custom-Scripts",
            "Comfyui-Memory_Cleanup",
            "ControlAltAI-Nodes",
            "comfyui-int-and-float",
            "Nvidia_RTX_Nodes_ComfyUI",
            "ComfyUI-VFI",
            "ComfyUI-LTXVideo",
        }
        self.assertTrue(required.issubset(names), required - names)

    def test_image_build_checks_workflow_nodes(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        checker = (ROOT / "scripts" / "check_workflow_nodes.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("check_workflow_nodes.py", dockerfile)
        self.assertIn("submodule update --init --recursive", installer)
        self.assertIn("download_rife.py", installer)
        self.assertIn("flownet.pkl", installer)
        self.assertIn('"Fast Groups Bypasser (rgthree)"', checker)
        self.assertIn('"easy getNode"', checker)
        self.assertIn('"easy setNode"', checker)
        self.assertIn("stub_gpu_only_imports", checker)
        self.assertIn("workflow node types missing from image", checker)

    def test_image_is_slimmed_and_starts_comfyui_immediately(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        build = (
            ROOT / ".github" / "workflows" / "build-ghcr.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("runpod/comfyui:1.4.4-cuda13.0@sha256:", dockerfile)
        self.assertIn("CUDA_FORCE_PRELOAD_LIBRARIES=0", dockerfile)
        self.assertIn("nvidia-modprobe", dockerfile)
        self.assertIn("repair_nvidia_devices", start)
        self.assertIn("cuInit", start)
        self.assertIn("claude-LTX image revision", start)
        self.assertIn("i2v.json original.json loop.json", start)
        self.assertIn("CLAUDE_LTX_REVISION", dockerfile)
        self.assertNotIn("wait_for_gpu", start)
        self.assertNotIn("GPU_WAIT_TIMEOUT", start)
        self.assertIn("claude-ltx:cuda13.0", build)
        self.assertIn("claude-ltx:cuda12.8", build)
        self.assertIn("claude-ltx:${{ github.sha }}", build)
        self.assertIn("BUILD_REVISION=${{ github.sha }}", build)
        self.assertLess(
            start.index("start_background_downloads"),
            start.index('exec "${PYTHON_BIN}" main.py'),
        )


if __name__ == "__main__":
    unittest.main()
