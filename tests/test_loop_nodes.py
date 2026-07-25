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
                "LTXMobiusSampler",
                "LTXLoopDecodeTiled",
                "LTXLoopAudioSeam",
            },
        )


if __name__ == "__main__":
    unittest.main()
