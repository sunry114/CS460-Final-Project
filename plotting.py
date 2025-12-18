from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

def plot_all(
    out_dir: Path,
    model_names: List[str],
    metrics: Dict[str, dict],
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    precision_score_thr: float,
) -> Tuple[Path, Path, Path]:
    ap50 = [metrics[n]["AP50"] for n in model_names]
    map_ = [metrics[n]["mAP_50_95"] for n in model_names]
    prec = [metrics[n]["Precision@0.5"] for n in model_names]
    miou = [metrics[n]["MeanIoU@0.5"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, ap50, width, label="AP@0.50")
    plt.bar(x, map_, width, label="mAP@0.50:0.95")
    plt.bar(x + width, prec, width, label=f"Precision (score>={precision_score_thr}) @ IoU>=0.50")
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("Traffic Light Detection Benchmark (BDD Video Subset, multi-frame)")
    plt.legend()
    p1 = out_dir / "bar_metrics_ap_map_precision.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(x, miou)
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title(f"Mean IoU over TPs (IoU>=0.50, score>={precision_score_thr})")
    p2 = out_dir / "bar_mean_iou.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for n in model_names:
        r, p = pr_curves[n]
        plt.plot(r, p, label=n)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Precision-Recall (IoU=0.50)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    p3 = out_dir / "pr_curve_iou50.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=200)
    plt.close()

    return p1, p2, p3
