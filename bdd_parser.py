import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def _extract_objects_from_frame(frame_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(frame_obj, dict):
        for k in ("labels", "objects", "annotations"):
            v = frame_obj.get(k)
            if isinstance(v, list):
                return v
    return []


def _extract_category(obj: Dict[str, Any]) -> str:
    for k in ("category", "label", "name", "type"):
        v = obj.get(k)
        if isinstance(v, str):
            return v
    return ""


def _extract_box2d(obj: Dict[str, Any]) -> Optional[List[float]]:
    if "box2d" in obj and isinstance(obj["box2d"], dict):
        b = obj["box2d"]
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]

    if "bbox" in obj:
        bb = obj["bbox"]
        if isinstance(bb, dict):
            if all(k in bb for k in ("x1", "y1", "x2", "y2")):
                return [float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])]
            if all(k in bb for k in ("x", "y", "w", "h")):
                x, y, w, h = float(bb["x"]), float(bb["y"]), float(bb["w"]), float(bb["h"])
                return [x, y, x + w, y + h]
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            bb = [float(x) for x in bb]
            if bb[2] > bb[0] and bb[3] > bb[1]:
                return bb
            x, y, w, h = bb
            return [x, y, x + w, y + h]

    if all(k in obj for k in ("x1", "y1", "x2", "y2")):
        return [float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])]

    return None


def _frame_index(frame_obj: Any, default_i: int) -> int:
    if isinstance(frame_obj, dict):
        for k in ("frameIndex", "frame_idx", "index", "id"):
            v = frame_obj.get(k)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    return default_i


def load_bdd_video_gt(json_path: Path, category_keywords: List[str]) -> Dict[int, List[List[float]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            return {}
        data = data[0]

    frames = []
    if isinstance(data, dict):
        if isinstance(data.get("frames"), list):
            frames = data["frames"]
        elif isinstance(data.get("labels"), list):
            frames = data["labels"]

    gt: Dict[int, List[List[float]]] = {}
    for i, fr in enumerate(frames):
        fi = _frame_index(fr, i)
        objs = _extract_objects_from_frame(fr)
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            cat = _extract_category(obj).lower().replace("_", " ").strip()
            if not any(kw in cat for kw in category_keywords):
                continue
            box = _extract_box2d(obj)
            if box is None:
                continue
            gt.setdefault(fi, []).append(box)
    return gt
