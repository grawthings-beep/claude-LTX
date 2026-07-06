import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import long_video  # noqa: E402


def sample_workflow():
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "example.png"},
            "_meta": {"title": "Load Image"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old prompt", "clip": ["9", 0]},
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "bad quality", "clip": ["9", 0]},
            "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
        },
        "4": {
            "class_type": "PrimitiveInt",
            "inputs": {"value": 121},
            "_meta": {"title": "number of frames"},
        },
        "5": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 42},
            "_meta": {"title": "RandomNoise"},
        },
        "6": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 43},
            "_meta": {"title": "RandomNoise refine"},
        },
        "8": {
            "class_type": "PrimitiveBoolean",
            "inputs": {"value": True},
            "_meta": {"title": "bypass_i2v"},
        },
    }


class PatchWorkflowTests(unittest.TestCase):
    def test_patch_image_prompt_frames_seed(self):
        patched = long_video.patch_workflow(
            sample_workflow(),
            image="frame_0001.png",
            prompt="new prompt",
            frames=241,
            seed=1000,
        )
        self.assertEqual(patched["1"]["inputs"]["image"], "frame_0001.png")
        self.assertEqual(patched["2"]["inputs"]["text"], "new prompt")
        self.assertEqual(patched["3"]["inputs"]["text"], "bad quality")
        self.assertEqual(patched["4"]["inputs"]["value"], 241)
        self.assertEqual(patched["5"]["inputs"]["noise_seed"], 1000)
        self.assertEqual(patched["6"]["inputs"]["noise_seed"], 1001)
        self.assertFalse(patched["8"]["inputs"]["value"],
                         "bypass_i2v must be forced off when an image is given")

    def test_bypass_i2v_untouched_without_image(self):
        patched = long_video.patch_workflow(sample_workflow(), prompt="x")
        self.assertTrue(patched["8"]["inputs"]["value"])

    def test_original_not_mutated(self):
        workflow = sample_workflow()
        long_video.patch_workflow(workflow, prompt="changed")
        self.assertEqual(workflow["2"]["inputs"]["text"], "old prompt")

    def test_invalid_frames_rejected(self):
        for frames in (0, 8, 100, 122):
            with self.assertRaises(ValueError):
                long_video.patch_workflow(sample_workflow(), frames=frames)

    def test_valid_frames_accepted(self):
        for frames in (9, 121, 241, 361, 481):
            long_video.validate_frames(frames)

    def test_missing_prompt_node_raises(self):
        workflow = sample_workflow()
        del workflow["2"]
        with self.assertRaises(ValueError):
            long_video.patch_workflow(workflow, prompt="x")

    def test_frames_fallback_to_latent_length(self):
        workflow = sample_workflow()
        del workflow["4"]
        workflow["7"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 960, "height": 544, "length": 121, "batch_size": 1},
            "_meta": {"title": "EmptyLTXVLatentVideo"},
        }
        patched = long_video.patch_workflow(workflow, frames=241)
        self.assertEqual(patched["7"]["inputs"]["length"], 241)


class HistoryParsingTests(unittest.TestCase):
    def test_find_video_outputs(self):
        entry = {
            "outputs": {
                "50": {
                    "images": [
                        {"filename": "preview.png", "subfolder": "", "type": "temp"}
                    ]
                },
                "51": {
                    "images": [
                        {"filename": "video_00001.mp4", "subfolder": "video",
                         "type": "output"}
                    ]
                },
            }
        }
        videos = long_video.find_video_outputs(entry)
        self.assertEqual(videos, [("video_00001.mp4", "video", "output")])

    def test_no_videos(self):
        self.assertEqual(long_video.find_video_outputs({"outputs": {}}), [])


if __name__ == "__main__":
    unittest.main()
