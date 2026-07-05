#!/usr/bin/env python3
import argparse
import pathlib
import sys


REQUIRED_CUSTOM_NODES = [
    "rgthree-comfy",
    "ComfyMath",
    "ComfyUI-VideoHelperSuite",
    "ComfyUI-LTXVideo",
    "ComfyUI-KJNodes",
]

REQUIRED_MODELS = [
    "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
    "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "models/loras/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()

    comfyui_dir = pathlib.Path(args.comfyui_dir)
    model_root = pathlib.Path(args.model_root)
    errors = []

    for name in REQUIRED_CUSTOM_NODES:
        if not (comfyui_dir / "custom_nodes" / name).exists():
            errors.append(f"missing custom node: {name}")

    for rel in REQUIRED_MODELS:
        path = model_root / rel
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing model: {rel}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Environment looks OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
