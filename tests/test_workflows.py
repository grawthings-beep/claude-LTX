import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
I2V_WORKFLOW = "mrxin-i2v.json"
HQ_I2V_WORKFLOW = "mrxin-i2v-hq.json"
AUTO_MOSAIC_WORKFLOW = "mrxin-i2v-auto-mosaic.json"
WORKFLOW_SHA256 = "635dfdb69b47eb9993313db2b1c4a4fdc0930b3b92bfce3b901c03352d4dc8f9"
HQ_WORKFLOW_SHA256 = "ef68769495a1acc50f0d9bd5d4bbbc354ca04affa4f19dc32027f3caf7f0e5be"
AUTO_MOSAIC_WORKFLOW_SHA256 = "2aa465caa8f330225a36cb39360d31fce9e5f79a71eb323125b9ef5fb0f161bf"


def load_workflow(name=I2V_WORKFLOW):
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


def assert_layout_is_packed(testcase, graph):
    def rectangle(values):
        x, y, width, height = map(float, values)
        return x, y, x + width, y + height

    def overlaps(left, right):
        return (
            min(left[2], right[2]) > max(left[0], right[0])
            and min(left[3], right[3]) > max(left[1], right[1])
        )

    def contains(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    def inflate(rect, amount):
        return (
            rect[0] - amount,
            rect[1] - amount,
            rect[2] + amount,
            rect[3] + amount,
        )

    groups = [rectangle(group["bounding"]) for group in graph["groups"]]
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            testcase.assertFalse(overlaps(left, right))
            testcase.assertFalse(overlaps(inflate(left, 24), inflate(right, 24)))

    node_rectangles = []
    for node in graph["nodes"]:
        node_rect = rectangle([*node["pos"], *node.get("size", [220, 80])[:2]])
        testcase.assertEqual(sum(contains(group, node_rect) for group in groups), 1)
        node_rectangles.append((node["id"], node_rect))
    for index, (left_id, left) in enumerate(node_rectangles):
        for right_id, right in node_rectangles[index + 1 :]:
            testcase.assertFalse(overlaps(left, right), (left_id, right_id))
            testcase.assertFalse(
                overlaps(inflate(left, 20), inflate(right, 20)),
                (left_id, right_id),
            )


class WorkflowTests(unittest.TestCase):
    def test_standard_hq_and_auto_mosaic_workflows_are_bundled(self):
        names = {path.name for path in WORKFLOWS.glob("*.json")}
        self.assertEqual(
            names, {I2V_WORKFLOW, HQ_I2V_WORKFLOW, AUTO_MOSAIC_WORKFLOW}
        )

    def test_noise_safe_workflow_is_locked(self):
        expected_hashes = {
            I2V_WORKFLOW: WORKFLOW_SHA256,
            HQ_I2V_WORKFLOW: HQ_WORKFLOW_SHA256,
            AUTO_MOSAIC_WORKFLOW: AUTO_MOSAIC_WORKFLOW_SHA256,
        }
        for name, expected_hash in expected_hashes.items():
            with self.subTest(workflow=name):
                canonical = json.dumps(
                    load_workflow(name), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                digest = hashlib.sha256(canonical).hexdigest()
                self.assertEqual(digest, expected_hash)

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
        self.assertEqual(nodes[18]["widgets_values"], [5, 5, 0])
        self.assertEqual(nodes[19]["widgets_values"], [960, 960, 0])
        self.assertEqual(nodes[181]["widgets_values"], [1280, 1280, 0])
        self.assertTrue(
            all(nodes[node_id]["widgets_values"] == [1] for node_id in (191, 192, 193, 194))
        )
        prompt = nodes[28]["widgets_values"][0]
        self.assertNotIn("3D, Real Video", prompt)
        self.assertNotIn("25 seconds", prompt)
        self.assertIn("Five seconds", prompt)

    def test_default_lora_stack_is_noise_safe(self):
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

    def test_resolution_slider_values_match_the_960x1280_default(self):
        nodes = {node["id"]: node for node in load_workflow()["nodes"]}

        for node_id, expected in ((19, 960), (181, 1280)):
            self.assertEqual(nodes[node_id]["properties"]["value"], expected)
            self.assertEqual(nodes[node_id]["widgets_values"][:2], [expected, expected])

    def test_hq_workflow_preserves_full_resolution_for_both_i2v_stages(self):
        workflow = load_workflow(HQ_I2V_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        subgraph = workflow["definitions"]["subgraphs"][0]
        subnodes = {node["id"]: node for node in subgraph["nodes"]}
        links = {link["id"]: link for link in subgraph["links"]}

        self.assertEqual(nodes[19]["properties"]["value"], 1792)
        self.assertEqual(nodes[19]["widgets_values"][:2], [1792, 1792])
        self.assertEqual(nodes[181]["properties"]["value"], 2368)
        self.assertEqual(nodes[181]["widgets_values"][:2], [2368, 2368])

        first_pass = (1792 // 2, 2368 // 2)
        self.assertEqual(first_pass, (896, 1184))
        self.assertTrue(all(value % 32 == 0 for value in first_pass))
        self.assertNotIn(
            "ResizeImagesByLongerEdge", {n["type"] for n in subnodes.values()}
        )
        self.assertEqual(links[379]["origin_id"], 178)
        self.assertEqual(links[379]["target_id"], 232)
        self.assertIn(379, subnodes[178]["outputs"][0]["links"])
        self.assertEqual(subnodes[232]["outputs"][0]["links"], [383, 384])
        self.assertEqual(subnodes[44]["inputs"][1]["link"], 383)
        self.assertEqual(subnodes[87]["inputs"][1]["link"], 384)

        self.assertEqual(len(workflow["nodes"]), 97)
        self.assertEqual(len(subgraph["nodes"]), 46)
        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, subgraph)

    def test_auto_mosaic_is_separate_and_runs_once_before_mp4_encode(self):
        for name in (I2V_WORKFLOW, HQ_I2V_WORKFLOW):
            self.assertNotIn(
                "WanAutoMosaicVideo",
                {node["type"] for node in load_workflow(name)["nodes"]},
            )

        workflow = load_workflow(AUTO_MOSAIC_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        mosaics = [
            node for node in workflow["nodes"] if node["type"] == "WanAutoMosaicVideo"
        ]
        self.assertEqual(len(mosaics), 1)
        mosaic = mosaics[0]
        self.assertEqual(
            mosaic["widgets_values"],
            [
                "ntd11_anime_nsfw_segm_v5.pt",
                "JUST",
                0.3,
                0.5,
                0,
                3,
                "pussy,penis,testicles",
            ],
        )

        incoming = links[mosaic["inputs"][0]["link"]]
        generator = nodes[incoming[1]]
        self.assertIn(
            generator["type"],
            {subgraph["id"] for subgraph in workflow["definitions"]["subgraphs"]},
        )
        self.assertEqual(incoming[2], 2)

        encoders = [
            node
            for node in workflow["nodes"]
            if node["type"] == "VHS_VideoCombine" and node.get("mode", 0) == 0
        ]
        self.assertEqual(len(encoders), 1)
        encoder = encoders[0]
        outgoing = links[encoder["inputs"][0]["link"]]
        self.assertEqual(outgoing[1:3], [mosaic["id"], 0])
        self.assertTrue(encoder["widgets_values"]["save_output"])
        self.assertEqual(
            links[encoder["inputs"][1]["link"]][1:3], [generator["id"], 3]
        )

    def test_auto_mosaic_uses_only_the_hq_first_pass_without_upscale(self):
        workflow = load_workflow(AUTO_MOSAIC_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        hq_nodes = {node["id"]: node for node in load_workflow(HQ_I2V_WORKFLOW)["nodes"]}
        subgraph = workflow["definitions"]["subgraphs"][0]
        subnodes = {node["id"]: node for node in subgraph["nodes"]}
        types = {node["type"] for node in workflow["nodes"] + subgraph["nodes"]}

        self.assertEqual(nodes[19]["properties"]["value"], 1792)
        self.assertEqual(nodes[181]["properties"]["value"], 2368)
        self.assertEqual(nodes[6]["widgets_values"], hq_nodes[6]["widgets_values"])
        self.assertEqual(subnodes[177]["widgets_values"], ["scale by multiplier", 0.5, "area"])
        self.assertEqual(
            workflow["extra"]["runpod_bundle"]["first_pass_resolution"],
            [896, 1184],
        )
        self.assertFalse(workflow["extra"]["runpod_bundle"]["latent_upscale"])
        self.assertNotIn("LatentUpscaleModelLoader", types)
        self.assertNotIn("LTXVLatentUpsampler", types)
        self.assertNotIn("ImageUpscaleWithModel", types)
        self.assertNotIn("UpscaleModelLoader", types)
        self.assertNotIn("RIFE VFI", types)
        self.assertEqual(
            [node["id"] for node in subgraph["nodes"] if node["type"] == "SamplerCustomAdvanced"],
            [51],
        )
        self.assertEqual(
            [output["label"] for output in subgraph["outputs"]],
            ["Resized Image", "FPS", "Video First Pass", "Audio First Pass"],
        )
        first_pass_output = next(
            link
            for link in subgraph["links"]
            if link["target_id"] == -20 and link["target_slot"] == 2
        )
        self.assertEqual(subnodes[first_pass_output["origin_id"]]["type"], "VAEDecode")

    def test_auto_mosaic_serialization_and_layout_are_complete(self):
        workflow = load_workflow(AUTO_MOSAIC_WORKFLOW)
        subgraph = workflow["definitions"]["subgraphs"][0]
        self.assertEqual(
            [group["id"] for group in workflow["groups"]],
            [52, 2, 44, 45, 46, 3, 4, 53, 5, 9, 6, 8, 49, 50, 25, 10, 7, 14, 16, 41],
        )
        self.assertFalse(
            any(
                node.get("title") == "Enable VIDEO EDITOR"
                for node in workflow["nodes"]
            )
        )
        assert_root_graph_complete(self, workflow)
        assert_subgraph_complete(self, subgraph)
        assert_layout_is_packed(self, workflow)
        assert_layout_is_packed(self, subgraph)

    def test_auto_mosaic_workflow_is_deterministically_generated(self):
        generator = (ROOT / "scripts" / "prepare_workflows.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_auto_mosaic_node", generator)
        self.assertIn("_prune_subgraph_to_first_pass", generator)
        self.assertIn("--check", generator)

    def test_i2v_conditioning_and_decode_defaults_reduce_artifacts(self):
        subgraph = load_workflow()["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in subgraph["nodes"]}

        self.assertEqual(nodes[44]["widgets_values"], [0.7, False])
        self.assertEqual(nodes[87]["widgets_values"], [1, False])
        self.assertEqual(nodes[149]["widgets_values"], [4, 4, 24, 4, True, "auto", "auto"])

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
