from typing import Any, Dict, List, Tuple
import numpy as np

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def voc_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    xs = np.linspace(0, 1, 101)
    ap = 0.0
    for x in xs:
        ap += np.max(mpre[mrec >= x]) if np.any(mrec >= x) else 0.0
    return float(ap / 101.0)


def evaluate_single_class(
    gt_by_image: Dict[int, List[np.ndarray]],
    dets: List[Tuple[int, np.ndarray, float]],  # (image_id, bbox_xyxy, score)
    iou_thresholds: List[float],
    precision_score_thr: float = 0.5,
    match_iou_for_stats: float = 0.5,
) -> Dict[str, Any]:
    num_gt = sum(len(v) for v in gt_by_image.values())
    dets_sorted = sorted(dets, key=lambda x: x[2], reverse=True)

    if num_gt == 0:
        return {
            "AP50": 0.0,
            "mAP_50_95": 0.0,
            "Precision@0.5": 0.0,
            "MeanIoU@0.5": 0.0,
            "AP_by_iou": {f"{t:.2f}": 0.0 for t in iou_thresholds},
            "PR_curve@0.5": (np.array([0.0]), np.array([0.0])),
            "NumGT": 0,
            "NumDet(>=min)": len(dets),
            "NumDet(>=0.5)": 0,
        }

    ap_by_iou: Dict[float, float] = {}
    pr_curve_50 = None

    for t in iou_thresholds:
        matched = {img_id: np.zeros((len(gts),), dtype=bool) for img_id, gts in gt_by_image.items()}
        tp = np.zeros((len(dets_sorted),), dtype=np.float32)
        fp = np.zeros((len(dets_sorted),), dtype=np.float32)

        for i, (img_id, bb, score) in enumerate(dets_sorted):
            gts = gt_by_image.get(img_id, [])
            if len(gts) == 0:
                fp[i] = 1.0
                continue
            ious = np.array([iou_xyxy(bb, gt) for gt in gts], dtype=np.float32)
            best = int(np.argmax(ious))
            best_iou = float(ious[best])
            if best_iou >= t and not matched[img_id][best]:
                tp[i] = 1.0
                matched[img_id][best] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(1.0, float(num_gt))
        precision = cum_tp / np.maximum(1.0, cum_tp + cum_fp)

        ap_by_iou[t] = voc_ap_101(recall, precision)
        if abs(t - 0.5) < 1e-9:
            pr_curve_50 = (recall.copy(), precision.copy())

    ap50 = ap_by_iou.get(0.5, 0.0)
    map_50_95 = float(np.mean([ap_by_iou[t] for t in iou_thresholds]))

    dets_thr = [(img, bb, sc) for (img, bb, sc) in dets if sc >= precision_score_thr]
    dets_thr = sorted(dets_thr, key=lambda x: x[2], reverse=True)

    matched_stats = {img_id: np.zeros((len(gts),), dtype=bool) for img_id, gts in gt_by_image.items()}
    TP, FP = 0, 0
    ious_tp: List[float] = []

    for img_id, bb, sc in dets_thr:
        gts = gt_by_image.get(img_id, [])
        if len(gts) == 0:
            FP += 1
            continue
        ious = np.array([iou_xyxy(bb, gt) for gt in gts], dtype=np.float32)
        best = int(np.argmax(ious))
        best_iou = float(ious[best])
        if best_iou >= match_iou_for_stats and not matched_stats[img_id][best]:
            TP += 1
            matched_stats[img_id][best] = True
            ious_tp.append(best_iou)
        else:
            FP += 1

    precision_at_thr = float(TP / max(1, TP + FP))
    mean_iou_tp = float(np.mean(ious_tp)) if len(ious_tp) > 0 else 0.0

    return {
        "AP50": float(ap50),
        "mAP_50_95": float(map_50_95),
        "Precision@0.5": precision_at_thr,
        "MeanIoU@0.5": mean_iou_tp,
        "AP_by_iou": {f"{t:.2f}": float(ap_by_iou[t]) for t in iou_thresholds},
        "PR_curve@0.5": pr_curve_50 if pr_curve_50 is not None else (np.array([0.0]), np.array([0.0])),
        "NumGT": int(num_gt),
        "NumDet(>=min)": int(len(dets)),
        "NumDet(>=0.5)": int(len(dets_thr)),
    }
