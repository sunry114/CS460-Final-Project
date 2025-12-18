import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm
import torch

from .bdd_parser import load_bdd_video_gt
from .detectors import ModelSpec, build_detectors
from .metrics import evaluate_single_class
from .plotting import plot_all
from .utils import make_json_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--ann_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_videos", type=int, default=0, help="0 = all")
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--precision_score_thr", type=float, default=0.5)
    parser.add_argument("--score_thr_ap_min", type=float, default=0.001)
    parser.add_argument("--target_kw", type=str, default="traffic light,traffic_light,trafficlight")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA not available. Using CPU.")
        device = "cpu"

    videos_dir = Path(args.videos_dir)
    ann_dir = Path(args.ann_dir)

    video_paths = sorted(list(videos_dir.rglob("*.mov")))
    if len(video_paths) == 0:
        raise FileNotFoundError(f"No .mov found under: {videos_dir}")

    if args.max_videos and args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]

    models = [
        ModelSpec(name="Our solution", kind="yolo", yolo_weight="yolov8m.pt", min_score_for_ap=args.score_thr_ap_min),
        ModelSpec(name="YOLOv8s", kind="yolo", yolo_weight="yolov8s.pt", min_score_for_ap=args.score_thr_ap_min),
        ModelSpec(name="FasterRCNN_R50FPN", kind="torchvision", min_score_for_ap=args.score_thr_ap_min),
    ]

    print("[Init] loading detectors ...")
    detectors = build_detectors(models, device=device)

    target_keywords = [x.strip().lower() for x in args.target_kw.split(",") if x.strip()]

    gt_by_image: Dict[int, List[np.ndarray]] = {}
    dets_by_model: Dict[str, List[Tuple[int, np.ndarray, float]]] = {m.name: [] for m in models}

    missing_labels: List[str] = []
    processed_videos = 0
    skipped_videos = 0
    global_img_id = 0

    print("[Run] per-frame detection (recursive .mov) ...")
    for vp in tqdm(video_paths, desc="Videos"):
        try:
            rel = vp.relative_to(videos_dir)
        except ValueError:
            rel = Path(vp.name)

        jp = (ann_dir / rel).with_suffix(".json")
        if not jp.exists():
            missing_labels.append(f"{vp} -> expected {jp}")
            skipped_videos += 1
            continue

        gt_frames = load_bdd_video_gt(jp, category_keywords=target_keywords)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            missing_labels.append(f"[Unreadable video] {vp}")
            skipped_videos += 1
            continue

        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if fi % max(1, args.frame_stride) != 0:
                fi += 1
                continue

            gt_boxes = gt_frames.get(fi, [])
            gt_by_image[global_img_id] = [np.array(b, dtype=np.float32) for b in gt_boxes]

            for m in models:
                det_arr = detectors[m.name].infer(frame)
                if det_arr.shape[0] == 0:
                    continue
                keep = det_arr[:, 4] >= float(m.min_score_for_ap)
                det_arr = det_arr[keep]
                for bb in det_arr:
                    dets_by_model[m.name].append((global_img_id, bb[:4].copy(), float(bb[4])))

            global_img_id += 1
            fi += 1

        cap.release()
        processed_videos += 1

    missing_path = out_dir / "missing_labels.txt"
    with open(missing_path, "w", encoding="utf-8") as f:
        f.write("\n".join(missing_labels))

    iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    metrics = {}
    pr_curves = {}

    print("[Eval] computing metrics ...")
    for m in models:
        res = evaluate_single_class(
            gt_by_image=gt_by_image,
            dets=dets_by_model[m.name],
            iou_thresholds=iou_thresholds,
            precision_score_thr=args.precision_score_thr,
            match_iou_for_stats=0.5,
        )
        metrics[m.name] = res
        pr_curves[m.name] = res["PR_curve@0.5"]

    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, indent=2, ensure_ascii=False)

    csv_lines = ["model,AP50,mAP_50_95,Precision@0.5,MeanIoU@0.5,NumGT,NumDet(>=min),NumDet(>=0.5)"]
    for name in [m.name for m in models]:
        r = metrics[name]
        csv_lines.append(
            f"{name},{r['AP50']:.4f},{r['mAP_50_95']:.4f},{r['Precision@0.5']:.4f},{r['MeanIoU@0.5']:.4f},"
            f"{r['NumGT']},{r['NumDet(>=min)']},{r['NumDet(>=0.5)']}"
        )
    csv_path = out_dir / "metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    model_names = [m.name for m in models]
    p1, p2, p3 = plot_all(out_dir, model_names, metrics, pr_curves, args.precision_score_thr)

    summary_path = out_dir / "run_summary.txt"
    summary_lines = [
        f"videos_dir: {videos_dir}",
        f"ann_dir: {ann_dir}",
        f"out_dir: {out_dir}",
        f"device: {device}",
        f"max_videos: {args.max_videos}",
        f"frame_stride: {args.frame_stride}",
        f"precision_score_thr: {args.precision_score_thr}",
        f"score_thr_ap_min: {args.score_thr_ap_min}",
        f"processed_videos: {processed_videos}",
        f"skipped_videos (missing/unreadable): {skipped_videos}",
        f"total_frames_evaluated: {len(gt_by_image)}",
        f"missing_labels_file: {missing_path}",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("[Done]")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {missing_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
