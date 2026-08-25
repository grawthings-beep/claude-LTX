#!/usr/bin/env python3
"""Generate the focused first-pass auto-mosaic workflow."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workflows" / "mrxin-i2v-hq.json"
OUTPUT = ROOT / "workflows" / "mrxin-i2v-auto-mosaic.json"
AUTO_WORKFLOW_ID = str(
    uuid.uuid5(uuid.NAMESPACE_URL, "claude-LTX/mrxin-i2v-auto-mosaic")
)
AUTO_SUBGRAPH_ID = str(
    uuid.uuid5(uuid.NAMESPACE_URL, "claude-LTX/mrxin-i2v-auto-mosaic/first-pass")
)


def _auto_mosaic_node(node_id, position, order):
    return {
        "id": node_id,
        "type": "WanAutoMosaicVideo",
        "pos": list(position),
        "size": [410, 360],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            {
                "name": "model_name",
                "type": "COMBO",
                "widget": {"name": "model_name"},
                "link": None,
            },
            {
                "name": "coverage_preset",
                "type": "COMBO",
                "widget": {"name": "coverage_preset"},
                "link": None,
            },
            {
                "name": "confidence",
                "type": "FLOAT",
                "widget": {"name": "confidence"},
                "link": None,
            },
            {
                "name": "iou_threshold",
                "type": "FLOAT",
                "widget": {"name": "iou_threshold"},
                "link": None,
            },
            {
                "name": "block_size",
                "type": "INT",
                "widget": {"name": "block_size"},
                "link": None,
            },
            {
                "name": "max_gap_frames",
                "type": "INT",
                "widget": {"name": "max_gap_frames"},
                "link": None,
            },
            {
                "name": "target_classes",
                "type": "STRING",
                "widget": {"name": "target_classes"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "mosaicked_images", "type": "IMAGE", "links": []}
        ],
        "properties": {"Node name for S&R": "WanAutoMosaicVideo"},
        "widgets_values": [
            "ntd11_anime_nsfw_segm_v5.pt",
            "JUST",
            0.30,
            0.50,
            0,
            3,
            "pussy,penis,testicles",
        ],
        "title": "AUTO MOSAIC JUST CONTOUR (CPU)",
    }


def _rebuild_root_endpoints(graph):
    by_id = {node["id"]: node for node in graph["nodes"]}
    for node in graph["nodes"]:
        for item in node.get("inputs", []):
            if "link" in item:
                item["link"] = None
        for item in node.get("outputs", []):
            if "links" in item:
                item["links"] = []

    for link_id, origin_id, origin_slot, target_id, target_slot, _type in graph["links"]:
        by_id[origin_id]["outputs"][origin_slot]["links"].append(link_id)
        by_id[target_id]["inputs"][target_slot]["link"] = link_id


def _rebuild_subgraph_endpoints(subgraph):
    by_id = {node["id"]: node for node in subgraph["nodes"]}
    for node in subgraph["nodes"]:
        for item in node.get("inputs", []):
            if "link" in item:
                item["link"] = None
        for item in node.get("outputs", []):
            if "links" in item:
                item["links"] = []

    for interface in subgraph["inputs"]:
        interface["linkIds"] = []
    for interface in subgraph["outputs"]:
        interface["linkIds"] = []

    for link in subgraph["links"]:
        link_id = link["id"]
        origin_id = link["origin_id"]
        target_id = link["target_id"]
        if origin_id == -10:
            subgraph["inputs"][link["origin_slot"]]["linkIds"].append(link_id)
        else:
            by_id[origin_id]["outputs"][link["origin_slot"]]["links"].append(
                link_id
            )
        if target_id == -20:
            subgraph["outputs"][link["target_slot"]]["linkIds"].append(link_id)
        else:
            by_id[target_id]["inputs"][link["target_slot"]]["link"] = link_id


def _flatten_and_pack_groups(graph):
    """Pack every node into one non-overlapping left-to-right group row."""
    groups = graph["groups"]

    def contains(group, node):
        gx, gy, gw, gh = map(float, group["bounding"])
        nx, ny = map(float, node["pos"])
        nw, nh = map(float, node.get("size", [220, 80])[:2])
        return (
            gx <= nx
            and gy <= ny
            and gx + gw >= nx + nw
            and gy + gh >= ny + nh
        )

    buckets = {}
    for node in graph["nodes"]:
        candidates = [group for group in groups if contains(group, node)]
        if not candidates:
            raise ValueError(f"root node is outside every group: {node['id']}")
        owner = min(
            candidates,
            key=lambda group: float(group["bounding"][2])
            * float(group["bounding"][3]),
        )
        buckets.setdefault(owner["id"], []).append(node)

    active_groups = [group for group in groups if group["id"] in buckets]
    active_groups.sort(
        key=lambda group: (float(group["bounding"][0]), float(group["bounding"][1]))
    )
    cursor_x = 0.0
    top = 3000.0
    for group in active_groups:
        nodes = sorted(
            buckets[group["id"]],
            key=lambda node: (float(node["pos"][1]), float(node["pos"][0])),
        )
        width = max(
            300.0,
            max(float(node.get("size", [220, 80])[0]) for node in nodes) + 20.0,
        )
        cursor_y = top + 55.0
        for node in nodes:
            node["pos"] = [cursor_x + 10.0, cursor_y]
            cursor_y += float(node.get("size", [220, 80])[1]) + 14.0
        group["bounding"] = [cursor_x, top, width, cursor_y - top + 10.0]
        cursor_x += width + 20.0

    graph["groups"] = active_groups
    graph.setdefault("extra", {})["ds"] = {
        "scale": 0.32,
        "offset": [120.0, -2700.0],
    }
    return cursor_x


def _prune_subgraph_to_first_pass(subgraph):
    links = {link["id"]: link for link in subgraph["links"]}
    inputs_by_target = {}
    for link in links.values():
        if link["target_id"] != -20:
            inputs_by_target.setdefault(link["target_id"], []).append(link)

    kept_output_slots = {0, 1, 2, 3}
    kept_link_ids = {
        link["id"]
        for link in links.values()
        if link["target_id"] == -20 and link["target_slot"] in kept_output_slots
    }
    pending = [
        links[link_id]["origin_id"]
        for link_id in kept_link_ids
        if links[link_id]["origin_id"] != -10
    ]
    kept_nodes = set()
    while pending:
        node_id = pending.pop()
        if node_id in kept_nodes:
            continue
        kept_nodes.add(node_id)
        for link in inputs_by_target.get(node_id, []):
            kept_link_ids.add(link["id"])
            if link["origin_id"] != -10:
                pending.append(link["origin_id"])

    kept_links = [
        copy.deepcopy(link)
        for link in subgraph["links"]
        if link["id"] in kept_link_ids
    ]
    used_input_slots = sorted(
        {link["origin_slot"] for link in kept_links if link["origin_id"] == -10}
    )
    input_slot_map = {
        old_slot: new_slot for new_slot, old_slot in enumerate(used_input_slots)
    }
    output_slot_map = {old_slot: old_slot for old_slot in sorted(kept_output_slots)}

    for link in kept_links:
        if link["origin_id"] == -10:
            link["origin_slot"] = input_slot_map[link["origin_slot"]]
        if link["target_id"] == -20:
            link["target_slot"] = output_slot_map[link["target_slot"]]

    subgraph["nodes"] = [
        node for node in subgraph["nodes"] if node["id"] in kept_nodes
    ]
    subgraph["links"] = kept_links
    subgraph["inputs"] = [
        subgraph["inputs"][old_slot] for old_slot in used_input_slots
    ]
    subgraph["outputs"] = [
        subgraph["outputs"][old_slot] for old_slot in sorted(kept_output_slots)
    ]
    subgraph["groups"] = [
        group
        for group in subgraph.get("groups", [])
        if group["id"] in {11, 12, 19, 21, 36, 37, 38}
    ]
    next(node for node in subgraph["nodes"] if node["id"] == 53)["pos"] = [
        2320,
        3500,
    ]
    right_edge = _flatten_and_pack_groups(subgraph)
    for index, interface in enumerate(subgraph["inputs"]):
        interface["pos"] = [-20.0, 3060.0 + index * 22.0]
    for index, interface in enumerate(subgraph["outputs"]):
        interface["pos"] = [right_edge + 20.0, 3060.0 + index * 22.0]
    subgraph["state"].update(
        {
            "lastGroupId": max(group["id"] for group in subgraph["groups"]),
            "lastNodeId": max(node["id"] for node in subgraph["nodes"]),
            "lastLinkId": max(link["id"] for link in subgraph["links"]),
        }
    )
    _rebuild_subgraph_endpoints(subgraph)
    return input_slot_map


def patch_auto_mosaic(source):
    graph = copy.deepcopy(source)
    graph["id"] = AUTO_WORKFLOW_ID
    graph["revision"] = 0

    subgraph = graph["definitions"]["subgraphs"][0]
    old_subgraph_id = subgraph["id"]
    input_slot_map = _prune_subgraph_to_first_pass(subgraph)
    subgraph["id"] = AUTO_SUBGRAPH_ID
    subgraph["name"] = "MrXin LTX 2.3 I2V First Pass"

    instance = next(node for node in graph["nodes"] if node["type"] == old_subgraph_id)
    instance["type"] = AUTO_SUBGRAPH_ID
    instance["inputs"] = [
        instance["inputs"][old_slot]
        for old_slot in sorted(input_slot_map, key=input_slot_map.get)
    ]
    instance["outputs"] = instance["outputs"][:4]
    instance["size"][1] = 286

    editor_bounds = next(
        group["bounding"] for group in graph["groups"] if group["id"] == 26
    )
    editor_x, _editor_y, editor_width, editor_height = map(float, editor_bounds)
    editor_y = 2600.0
    editor_height += 455.0
    editor_node_ids = {
        node["id"]
        for node in graph["nodes"]
        if editor_x <= float(node["pos"][0]) <= editor_x + editor_width
        and editor_y <= float(node["pos"][1]) <= editor_y + editor_height
    }
    removed_node_ids = editor_node_ids | {4, 13, 110, 226, 228, 230, 61}
    graph["nodes"] = [
        node for node in graph["nodes"] if node["id"] not in removed_node_ids
    ]
    graph["groups"] = [
        group
        for group in graph["groups"]
        if group["id"] not in {26, 27, 28, 29, 32, 34, 42, 51}
    ]
    graph["links"] = [
        link
        for link in graph["links"]
        if link[1] not in removed_node_ids and link[3] not in removed_node_ids
    ]

    for link in graph["links"]:
        if link[3] == instance["id"]:
            link[4] = input_slot_map[link[4]]

    encoder = next(node for node in graph["nodes"] if node["id"] == 59)
    encoder["pos"] = [3140, 3200]
    encoder["title"] = "AUTO MOSAIC MP4"
    encoder["widgets_values"].update(
        {
            "filename_prefix": "MrXin/LTX2.3/AutoMosaic/FirstPass",
            "save_metadata": True,
            "save_output": True,
            "videopreview": {"hidden": False, "paused": False, "params": {}},
        }
    )

    mosaic_id = max(node["id"] for node in graph["nodes"]) + 1
    mosaic = _auto_mosaic_node(
        mosaic_id,
        (2679, 3200),
        max(node.get("order", 0) for node in graph["nodes"]) + 1,
    )
    graph["nodes"].append(mosaic)

    upstream_link = next(link for link in graph["links"] if link[0] == 360)
    if upstream_link[1:3] != [instance["id"], 2] or upstream_link[3] != encoder["id"]:
        raise ValueError("unexpected first-pass video wiring in source workflow")
    upstream_link[3] = mosaic_id
    upstream_link[4] = 0
    mosaic_to_encode = max(link[0] for link in graph["links"]) + 1
    graph["links"].append(
        [mosaic_to_encode, mosaic_id, 0, encoder["id"], 0, "IMAGE"]
    )

    parent = next(group for group in graph["groups"] if group["id"] == 25)
    parent["bounding"][2] = 4549
    parent["bounding"][3] = 1155
    mosaic_group = next(group for group in graph["groups"] if group["id"] == 14)
    mosaic_group.update(
        {
            "title": "Auto Mosaic (CPU / JUST)",
            "bounding": [2669, 3101, 430, 500],
            "color": "#7a3f83",
        }
    )
    output_group = next(group for group in graph["groups"] if group["id"] == 16)
    output_group.update(
        {
            "title": "Auto Mosaic Video",
            "bounding": [3129, 3101, 300, 978],
        }
    )
    cleanup_group = next(group for group in graph["groups"] if group["id"] == 41)
    cleanup_group["bounding"][3] = 210
    graph["groups"].extend(
        [
            {
                "id": 52,
                "title": "Workflow Notes",
                "bounding": [-1660, 2600, 520, 1505],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 53,
                "title": "Workflow Controls",
                "bounding": [-1110, 2850, 4540, 155],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 54,
                "title": "Resolution Notes",
                "bounding": [890, 2600, 1150, 220],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
        ]
    )

    _flatten_and_pack_groups(graph)

    graph["last_node_id"] = max(node["id"] for node in graph["nodes"])
    graph["last_link_id"] = max(link[0] for link in graph["links"])
    graph.setdefault("extra", {})["runpod_bundle"] = {
        "preset": "mrxin-i2v-first-pass-auto-mosaic",
        "postprocess": (
            "Anime NSFW Detection v5 YOLO11-seg JUST contour mosaic "
            "on CPU before MP4 encode"
        ),
        "requires": [
            "models/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt",
            "CIVITAI_API_TOKEN",
        ],
        "first_pass_resolution": [896, 1184],
        "latent_upscale": False,
    }
    _rebuild_root_endpoints(graph)
    return graph


def encode(graph):
    return (
        json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected = encode(patch_auto_mosaic(source))
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != expected:
            print(f"Generated workflow is stale: {OUTPUT.relative_to(ROOT)}", file=sys.stderr)
            return 1
        return 0

    OUTPUT.write_bytes(expected)
    print(f"WROTE {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
