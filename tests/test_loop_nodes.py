import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "custom_nodepacks" / "ComfyUI-LTXLoop" / "nodes.py"


def load_nodes_without_torch():
    torch = types.ModuleType("torch")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    k_diffusion = types.ModuleType("comfy.k_diffusion")
    k_diffusion.__path__ = []
    sampling = types.ModuleType("comfy.k_diffusion.sampling")
    samplers = types.ModuleType("comfy.samplers")
    comfy.k_diffusion = k_diffusion
    comfy.samplers = samplers
    k_diffusion.sampling = sampling

    stubs = {
        "torch": torch,
        "comfy": comfy,
        "comfy.k_diffusion": k_diffusion,
        "comfy.k_diffusion.sampling": sampling,
        "comfy.samplers": samplers,
    }
    spec = importlib.util.spec_from_file_location("ltx_loop_nodes_test", NODE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class LoopNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_nodes_without_torch()

    def test_latent_shape_matches_1728x1152_loop(self):
        self.assertEqual(
            self.nodes.latent_video_shape(1728, 1152, 161),
            (128, 21, 36, 54),
        )

    def test_latent_shape_rejects_non_ltx_dimensions(self):
        with self.assertRaises(ValueError):
            self.nodes.latent_video_shape(1700, 1152, 161)
        with self.assertRaises(ValueError):
            self.nodes.latent_video_shape(1728, 1152, 1)

    def test_all_workflow_nodes_are_registered(self):
        self.assertEqual(
            set(self.nodes.NODE_CLASS_MAPPINGS),
            {
                "SetImageSize",
                "LTXSetImageSize",
                "LTXLoopBridgeFrames",
                "LTXLoopAssemble",
                "LTXMobiusSampler",
                "LTXLoopDecodeTiled",
                "LTXLoopAudioSeam",
            },
        )

    def test_image_size_node_preserves_dimensions(self):
        self.assertEqual(
            self.nodes.LTXSetImageSize().size(1728, 1152),
            (1728, 1152),
        )
        self.assertIs(
            self.nodes.NODE_CLASS_MAPPINGS["SetImageSize"],
            self.nodes.LTXSetImageSize,
        )

    def test_bridge_plan_keeps_six_second_cycle_without_duplicate_contexts(self):
        self.assertEqual(
            self.nodes.bridge_frame_plan(129, 49, 9),
            (120, 40, 160),
        )

    def test_bridge_plan_rejects_invalid_contexts(self):
        with self.assertRaises(ValueError):
            self.nodes.bridge_frame_plan(129, 49, 8)
        with self.assertRaises(ValueError):
            self.nodes.bridge_frame_plan(9, 49, 9)
        with self.assertRaises(ValueError):
            self.nodes.bridge_frame_plan(129, 9, 9)

    def test_audio_trim_matches_kept_video_duration(self):
        waveform = types.SimpleNamespace(shape=(1, 2, 500000))
        self.assertEqual(
            self.nodes._audio_keep_samples(waveform, 120, 44100, 24),
            220500,
        )


if __name__ == "__main__":
    unittest.main()
