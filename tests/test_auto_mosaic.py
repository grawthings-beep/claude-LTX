import importlib.util
import inspect
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "custom_nodepacks" / "ComfyUI-LTXLoop" / "mosaic_nodes.py"
)
PACKAGE_INIT = MODULE_PATH.parent / "__init__.py"


def load_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "/workspace/comfyui/models"
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("ltx_mosaic_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_package_without_torch():
    torch = types.ModuleType("torch")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    k_diffusion = types.ModuleType("comfy.k_diffusion")
    k_diffusion.__path__ = []
    sampling = types.ModuleType("comfy.k_diffusion.sampling")
    samplers = types.ModuleType("comfy.samplers")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "/workspace/comfyui/models"
    comfy.k_diffusion = k_diffusion
    comfy.samplers = samplers
    k_diffusion.sampling = sampling
    stubs = {
        "torch": torch,
        "comfy": comfy,
        "comfy.k_diffusion": k_diffusion,
        "comfy.k_diffusion.sampling": sampling,
        "comfy.samplers": samplers,
        "folder_paths": folder_paths,
    }
    spec = importlib.util.spec_from_file_location(
        "ltx_loop_package_under_test",
        PACKAGE_INIT,
        submodule_search_locations=[str(PACKAGE_INIT.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class AutoMosaicHelperTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_box_expansion_is_clamped_to_frame(self):
        self.assertEqual(
            self.module._expanded_box((5, 10, 25, 30), 30, 35, 1.0),
            (0, 0, 30, 35),
        )

    def test_temporal_gap_fill_preserves_detected_contours_and_wraps_seam(self):
        first = np.zeros((3, 3), dtype=np.bool_)
        third = np.zeros((3, 3), dtype=np.bool_)
        first[1, 0] = True
        third[1, 2] = True
        filled = self.module._fill_short_circular_gaps(
            [first, None, third, None], 1
        )
        self.assertTrue(np.array_equal(filled[0], first))
        self.assertTrue(np.array_equal(filled[2], third))
        self.assertIsNotNone(filled[1])
        self.assertIsNotNone(filled[3])

    def test_gap_larger_than_limit_is_not_filled(self):
        first = np.zeros((3, 3), dtype=np.bool_)
        last = np.zeros((3, 3), dtype=np.bool_)
        first[1, 0] = True
        last[1, 2] = True
        filled = self.module._fill_short_circular_gaps(
            [first, None, None, last], 1
        )
        self.assertIsNone(filled[1])
        self.assertIsNone(filled[2])

    def test_fixed_grid_mosaic_uses_stationary_blocks(self):
        image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        result = self.module._fixed_grid_mosaic(image, 3)
        self.assertTrue(np.all(result[0:3, 0:3] == result[0, 0]))
        self.assertTrue(np.all(result[3:6, 3:6] == result[3, 3]))
        self.assertFalse(np.array_equal(result[0, 0], result[3, 3]))

    def test_default_targets_exclude_anus_and_large_context_classes(self):
        ids = self.module._selected_class_ids(
            {
                0: "nipples",
                1: "pussy",
                2: "anus",
                3: "penis",
                4: "cross-section",
                5: "x-ray",
                6: "testicles",
            },
            self.module.DEFAULT_CLASSES,
        )
        self.assertEqual(ids, [1, 3, 6])
        self.assertNotIn("anus", self.module.DEFAULT_CLASSES)

    def test_auto_block_size_uses_short_side_rule(self):
        self.assertEqual(self.module._resolve_block_size(0, 896, 1184), 18)
        self.assertEqual(self.module._resolve_block_size(36, 896, 1184), 36)

    def test_node_defaults_are_just_contour_settings(self):
        required = self.module.WanAutoMosaicVideo.INPUT_TYPES()["required"]
        self.assertEqual(list(self.module.COVERAGE_PRESETS)[0], "JUST")
        self.assertFalse(self.module.COVERAGE_PRESETS["JUST"]["ellipse"])
        self.assertEqual(required["confidence"][1]["default"], 0.30)
        self.assertEqual(required["iou_threshold"][1]["default"], 0.50)
        self.assertEqual(required["block_size"][1]["default"], 0)
        self.assertEqual(required["max_gap_frames"][1]["default"], 3)
        self.assertEqual(
            required["target_classes"][1]["default"],
            "pussy,penis,testicles",
        )

    def test_inference_is_forced_to_cpu(self):
        source = inspect.getsource(self.module.WanAutoMosaicVideo.apply)
        self.assertIn('device="cpu"', source)
        self.assertIn("half=False", source)
        self.assertIn("retina_masks=True", source)

    def test_custom_node_mapping_is_registered(self):
        self.assertIs(
            self.module.NODE_CLASS_MAPPINGS["WanAutoMosaicVideo"],
            self.module.WanAutoMosaicVideo,
        )

    def test_package_merges_loop_and_mosaic_node_mappings(self):
        package = load_package_without_torch()
        self.assertIn("LTXLoopPhaseCut", package.NODE_CLASS_MAPPINGS)
        self.assertIn("WanAutoMosaicVideo", package.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
