#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import pathlib
import sys


FRONTEND_NODE_TYPES = {"MarkdownNote", "Note"}


def workflow_node_types(path):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    subgraph_ids = {subgraph["id"] for subgraph in subgraphs}
    types = {node["type"] for node in workflow.get("nodes", [])}
    for subgraph in subgraphs:
        types.update(node["type"] for node in subgraph.get("nodes", []))
    return types - subgraph_ids - FRONTEND_NODE_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", required=True)
    parser.add_argument("--workflows", required=True)
    args = parser.parse_args()

    comfyui_dir = pathlib.Path(args.comfyui_dir).resolve()
    workflow_dir = pathlib.Path(args.workflows).resolve()
    os.chdir(comfyui_dir)
    sys.path.insert(0, str(comfyui_dir))
    sys.argv = [sys.argv[0], "--cpu"]

    import comfy.options

    comfy.options.enable_args_parsing()
    import utils.install_util
    import server
    import nodes

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop)
    try:
        loop.run_until_complete(nodes.init_extra_nodes(init_api_nodes=False))
    finally:
        loop.close()

    required = set()
    for path in workflow_dir.glob("*.json"):
        required.update(workflow_node_types(path))
    missing = required - set(nodes.NODE_CLASS_MAPPINGS)
    if missing:
        raise SystemExit(
            "workflow node types missing from image: " + ", ".join(sorted(missing))
        )
    print(f"Workflow node smoke test passed ({len(required)} node types).")


if __name__ == "__main__":
    main()
