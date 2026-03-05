# Copyright (c) TIER IV, Inc. All rights reserved.

"""3D semantic segmentation metrics.

Architecture:
  MetricsResult     - holds per-class arrays + class-avg / point-avg scalars
  compute_metrics() - creates MetricsResult from (intersection, union, target)
  format_*()        - plain-text and table formatters (no I/O)
  CountsAccumulator - accumulates counts across batches
  T4SegMetric       - MMEngine BaseMetric wrapper for FRNet
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from autoware_ml.registry import METRICS

_EPS = 1e-10
METRIC_NAMES = ("iou", "accuracy", "precision", "recall", "f1")
_METRIC_SHORT = {"iou": "IoU", "accuracy": "Acc", "precision": "Prec", "recall": "Rec", "f1": "F1"}


class MetricsResult:
    """Segmentation metrics computed from (intersection, union, target) counts.

    Attributes:
        per_class: metric name -> np.ndarray of length num_classes.
        class_avg: metric name -> float (mean over classes).
        point_avg: metric name -> float (pooled over all points).
    """

    __slots__ = ("per_class", "class_avg", "point_avg")

    def __init__(
        self,
        per_class: Dict[str, np.ndarray],
        class_avg: Dict[str, float],
        point_avg: Dict[str, float],
    ):
        self.per_class = per_class
        self.class_avg = class_avg
        self.point_avg = point_avg

    @property
    def num_classes(self) -> int:
        return len(next(iter(self.per_class.values())))

    def to_flat_dict(
        self,
        prefix: str = "",
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Export as flat dict for TensorBoard / MMEngine."""
        out: Dict[str, Union[float, np.ndarray]] = {}
        for m in METRIC_NAMES:
            out[f"{prefix}{m}_class"] = self.class_avg[m]
            out[f"{prefix}{m}_points"] = self.point_avg[m]
            out[f"{prefix}{m}_per_class"] = self.per_class[m]
        if class_names and len(class_names) == self.num_classes:
            for i, name in enumerate(class_names):
                safe = name.replace(" ", "_")
                for m in METRIC_NAMES:
                    out[f"{prefix}class_{i}_{safe}_{m}"] = float(self.per_class[m][i])
        return out


def compute_metrics(
    intersection: np.ndarray,
    union: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    eps: float = _EPS,
) -> MetricsResult:
    """Compute all segmentation metrics from aggregated counts.

    This is the single source of truth used by both PTv3 and FRNet.
    """
    inter = np.asarray(intersection, dtype=np.float64).ravel()
    un = np.asarray(union, dtype=np.float64).ravel()
    tgt = np.asarray(target, dtype=np.float64).ravel()
    if inter.size != num_classes or un.size != num_classes or tgt.size != num_classes:
        raise ValueError(
            f"intersection/union/target size must match num_classes={num_classes}, "
            f"got {inter.size}, {un.size}, {tgt.size}"
        )

    area_pred = np.maximum(un - tgt + inter, 0.0)

    iou_arr = inter / (un + eps)
    acc_arr = inter / (tgt + eps)
    prec_arr = inter / (area_pred + eps)
    rec_arr = acc_arr
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2.0 * prec_arr * rec_arr / (prec_arr + rec_arr + eps),
        0.0,
    )

    per_class = dict(iou=iou_arr, accuracy=acc_arr, precision=prec_arr, recall=rec_arr, f1=f1_arr)
    class_avg = {m: float(np.mean(per_class[m])) for m in METRIC_NAMES}

    total_tp = float(np.sum(inter))
    total_pred = float(np.sum(area_pred))
    total_gt = float(np.sum(tgt))
    total_un = float(np.sum(un))
    prec_pts = total_tp / (total_pred + eps)
    rec_pts = total_tp / (total_gt + eps)
    f1_pts = 2.0 * prec_pts * rec_pts / (prec_pts + rec_pts + eps) if (prec_pts + rec_pts) > 0 else 0.0
    point_avg = dict(
        iou=total_tp / (total_un + eps),
        accuracy=rec_pts,
        precision=prec_pts,
        recall=rec_pts,
        f1=f1_pts,
    )

    return MetricsResult(per_class, class_avg, point_avg)


def compute_metrics_from_counts(
    intersection: np.ndarray,
    union: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
    class_names: Optional[List[str]] = None,
    prefix: str = "",
    eps: float = _EPS,
) -> Dict[str, Union[float, np.ndarray]]:
    """Legacy wrapper: returns a flat dict. Prefer compute_metrics() for new code."""
    return compute_metrics(intersection, union, target, num_classes, eps).to_flat_dict(prefix, class_names)


def scalar_metrics(metrics: Dict[str, Union[float, np.ndarray]]) -> Dict[str, float]:
    """Filter a flat metrics dict to scalar-only entries."""
    return {
        k: float(v) if isinstance(v, np.floating) else v
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.floating))
    }


def _format_table(headers: list, rows: list, val_width: int = 7) -> str:
    """Generic ASCII table. First column left-aligned, rest right-aligned."""
    first_w = max(len(str(headers[0])), *(len(str(r[0])) for r in rows))
    sep = " | "

    def _row(label, vals):
        parts = [f"{str(label):<{first_w}}"]
        for v in vals:
            if isinstance(v, (int, float, np.floating)):
                parts.append(f"{float(v):>{val_width}.4f}")
            else:
                parts.append(f"{str(v):>{val_width}}")
        return sep.join(parts)

    lines = [_row(headers[0], headers[1:])]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(_row(row[0], row[1:]))
    return "\n".join(lines)


def format_summary(result: MetricsResult, label: str = "") -> List[str]:
    """Two compact lines: class-avg and point-avg."""
    prefix = f"{label} " if label else ""
    cls_parts = [f"{_METRIC_SHORT[m]}={result.class_avg[m]:.4f}" for m in METRIC_NAMES]
    pts_parts = [f"{_METRIC_SHORT[m]}={result.point_avg[m]:.4f}" for m in METRIC_NAMES]
    return [
        f"{prefix}class-avg: {' | '.join(cls_parts)}",
        f"{prefix}point-avg: {' | '.join(pts_parts)}",
    ]


def format_class_table(result: MetricsResult, class_names: List[str]) -> str:
    """Table: one row per class, columns = metrics, with summary rows."""
    headers = ["Class"] + [_METRIC_SHORT[m] for m in METRIC_NAMES]
    rows = []
    for i, name in enumerate(class_names):
        rows.append([name] + [float(result.per_class[m][i]) for m in METRIC_NAMES])
    rows.append(["--- class-avg"] + [result.class_avg[m] for m in METRIC_NAMES])
    rows.append(["--- point-avg"] + [result.point_avg[m] for m in METRIC_NAMES])
    return _format_table(headers, rows)


def format_range_class_table(
    results: "OrderedDict[str, MetricsResult]",
    metric_name: str,
    class_names: List[str],
) -> str:
    """Table: rows = classes, columns = range buckets, for one metric.

    Args:
        results: Ordered mapping label -> MetricsResult (e.g. {"Full": ..., "0-20m": ...}).
        metric_name: One of METRIC_NAMES.
        class_names: List of class names matching num_classes.
    """
    labels = list(results.keys())
    headers = [_METRIC_SHORT[metric_name]] + labels
    rows = []
    for i, name in enumerate(class_names):
        rows.append([name] + [float(results[lbl].per_class[metric_name][i]) for lbl in labels])
    rows.append(["--- class-avg"] + [results[lbl].class_avg[metric_name] for lbl in labels])
    rows.append(["--- point-avg"] + [results[lbl].point_avg[metric_name] for lbl in labels])
    return _format_table(headers, rows)


def format_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    label: str = "",
) -> str:
    """Format confusion matrix as a compact ASCII table.

    Args:
        cm: (num_classes, num_classes) array, cm[gt][pred] = count.
        class_names: Human-readable class names.
        normalize: If True, row-normalize and show percentages (0-100).
        label: Optional label appended to the title (e.g. "0-20m").
    """
    nc = cm.shape[0]
    cm_show = normalize_confusion_matrix(cm) * 100.0 if normalize else cm.copy()
    col_w = 6
    trunc = [n.replace(" ", "_")[:col_w] for n in class_names]
    label_w = max(len(n) for n in class_names) + 2

    base = "Confusion Matrix"
    if label:
        base += f" [{label}]"
    title = f"{base} (row-normalized %)" if normalize else f"{base} (counts)"
    gt_pred = "GT \\ Pred"
    header = f"{gt_pred:<{label_w}}" + " ".join(f"{t:>{col_w}}" for t in trunc)
    sep = "-" * len(header)

    lines = [title, header, sep]
    for i in range(nc):
        cells = " ".join(
            f"{cm_show[i, j]:>{col_w}.1f}" if normalize else f"{int(cm_show[i, j]):>{col_w}d}" for j in range(nc)
        )
        lines.append(f"{class_names[i]:<{label_w}}{cells}")
    return "\n".join(lines)


def confusion_matrix_to_flat_dict(
    cm: np.ndarray,
    class_names: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    """Export row-normalized CM as flat dict for TensorBoard scalars.

    Tags: ``{prefix}cm/{gt_name}/pred_{pred_name}`` with value in [0, 1].
    """
    nc = cm.shape[0]
    cm_norm = normalize_confusion_matrix(cm)
    out: Dict[str, float] = {}
    for i in range(nc):
        gt = class_names[i].replace(" ", "_")
        for j in range(nc):
            pred_name = class_names[j].replace(" ", "_")
            out[f"{prefix}cm/{gt}/pred_{pred_name}"] = float(cm_norm[i, j])
    return out


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    label: str = "",
):
    """Render confusion matrix as a matplotlib Figure for TensorBoard.

    Y-axis = "True label", X-axis = "Predicted label".
    Color scale is fixed 0-1 (normalized) so plots are comparable across epochs.
    Cell values are printed inside each cell.

    Returns:
        matplotlib.figure.Figure (caller should close after use).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as MplNormalize

    nc = cm.shape[0]
    cm_norm = normalize_confusion_matrix(cm)

    fig, ax = plt.subplots(figsize=(max(10, nc * 0.55), max(8, nc * 0.5)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", norm=MplNormalize(vmin=0.0, vmax=1.0))
    fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(nc):
        for j in range(nc):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=max(4, 7 - nc // 10), color=color)

    title = "Confusion Matrix"
    if label:
        title += f" [{label}]"
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)

    tick_marks = np.arange(nc)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    fig.tight_layout()
    return fig


def range_label(lo: float, hi: float) -> str:
    """Human-readable range label, e.g. '0-20m'."""
    lo_s = f"{lo:g}"
    hi_s = f"{hi:g}"
    return f"{lo_s}-{hi_s}m"


def intersection_union_target_np(
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class intersection, union, target from flat pred/label."""
    pred = pred.ravel().copy()
    label = label.ravel()
    pred[label == ignore_index] = ignore_index
    valid = (pred >= 0) & (pred < num_classes) & (label >= 0) & (label < num_classes)
    area_intersection = np.bincount(pred[valid], minlength=num_classes).astype(np.float64)
    area_pred = np.bincount(pred[pred >= 0], minlength=num_classes).astype(np.float64)[:num_classes]
    area_target = np.bincount(label[label >= 0], minlength=num_classes).astype(np.float64)[:num_classes]
    return area_intersection, area_pred + area_target - area_intersection, area_target


def bev_distance(coord: np.ndarray) -> np.ndarray:
    """BEV radius from ego: sqrt(x^2 + y^2)."""
    return np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)


def confusion_matrix_np(
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Compute confusion matrix where ``cm[gt][pred]`` = point count.

    Returns:
        np.ndarray of shape (num_classes, num_classes).
    """
    pred = pred.ravel()
    label = label.ravel()
    valid = (label != ignore_index) & (label >= 0) & (label < num_classes) & (pred >= 0) & (pred < num_classes)
    indices = label[valid] * num_classes + pred[valid]
    return np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes).astype(np.float64)


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Row-normalize so each row sums to 1 (GT-class perspective)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return cm / safe_sums


class CountsAccumulator:
    """Accumulates per-class intersection/union/target for named buckets.

    Usage::

        acc = CountsAccumulator(num_classes)
        acc.add("full", intersection, union, target)
        acc.add_masked("range_0_20", pred, gt, mask, ...)
        for name, (i, u, t) in acc.items():
            result = compute_metrics(i, u, t, num_classes)
    """

    def __init__(self, num_classes: int):
        self._nc = num_classes
        self._data: Dict[str, List[np.ndarray]] = {}

    def _ensure(self, key: str) -> List[np.ndarray]:
        if key not in self._data:
            self._data[key] = [np.zeros(self._nc, dtype=np.float64) for _ in range(3)]
        return self._data[key]

    def add(self, key: str, inter: np.ndarray, union: np.ndarray, target: np.ndarray) -> None:
        buf = self._ensure(key)
        buf[0] += inter
        buf[1] += union
        buf[2] += target

    def add_masked(
        self,
        key: str,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        num_classes: int,
        ignore_index: int,
    ) -> None:
        if not np.any(mask):
            return
        i, u, t = intersection_union_target_np(pred[mask], gt[mask], num_classes, ignore_index)
        self.add(key, i, u, t)

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (intersection, union, target) for *key*, or None."""
        if key in self._data:
            return tuple(self._data[key])
        return None

    def items(self):
        for key, (i, u, t) in self._data.items():
            yield key, i, u, t

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __bool__(self) -> bool:
        return bool(self._data)


class SegEvaluationReport:
    """Precomputed segmentation evaluation results.

    Created via :meth:`from_counts`, then consumed via :meth:`log`,
    :meth:`to_flat_dict`, or :meth:`write_tensorboard`.  This single class
    replaces ad-hoc formatting/export code that was previously duplicated in
    PTv3's evaluator hook, PTv3's test script, and T4SegMetric.
    """

    __slots__ = ("full", "cm", "range_entries")

    def __init__(
        self,
        full: MetricsResult,
        cm: np.ndarray,
        range_entries: Optional[List[Tuple[float, float, MetricsResult, Optional[np.ndarray]]]] = None,
    ):
        self.full = full
        self.cm = cm
        self.range_entries = range_entries or []

    @classmethod
    def from_counts(
        cls,
        intersection: np.ndarray,
        union: np.ndarray,
        target: np.ndarray,
        num_classes: int,
        cm: np.ndarray,
        range_acc: Optional[CountsAccumulator] = None,
        range_cms: Optional[Dict[str, np.ndarray]] = None,
        distance_ranges: Optional[List[Tuple[float, float]]] = None,
    ) -> "SegEvaluationReport":
        """Build a report from accumulated counts."""
        full = compute_metrics(intersection, union, target, num_classes)
        entries: List[Tuple[float, float, MetricsResult, Optional[np.ndarray]]] = []
        if distance_ranges and range_acc:
            for lo, hi in distance_ranges:
                key = f"range_{lo}_{hi}"
                counts = range_acc.get(key)
                if counts is not None:
                    ri, ru, rt = counts
                    result = compute_metrics(ri, ru, rt, num_classes)
                    rcm = range_cms.get(key) if range_cms else None
                    entries.append((lo, hi, result, rcm))
        return cls(full, cm, entries)

    def log(self, logger, class_names: List[str], label: str = "Val") -> None:
        """Format and log all metrics, tables, and confusion matrices."""
        for line in format_summary(self.full, label):
            logger.info(line)
        logger.info("\n" + format_class_table(self.full, class_names))
        logger.info("\n" + format_confusion_matrix(self.cm, class_names))

        if self.range_entries:
            range_results = OrderedDict({"Full": self.full})
            for lo, hi, result, _ in self.range_entries:
                range_results[range_label(lo, hi)] = result

            for m in METRIC_NAMES:
                logger.info("\n" + format_range_class_table(range_results, m, class_names))

            for lo, hi, _, rcm in self.range_entries:
                if rcm is not None:
                    logger.info(
                        "\n"
                        + format_confusion_matrix(
                            rcm,
                            class_names,
                            label=range_label(lo, hi),
                        )
                    )

    def to_flat_dict(
        self,
        class_names: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Export all metrics as a flat scalar dict for MMEngine."""
        out: Dict[str, float] = {}
        out.update(scalar_metrics(self.full.to_flat_dict(prefix, class_names)))
        if class_names:
            out.update(confusion_matrix_to_flat_dict(self.cm, class_names, prefix))
        for lo, hi, result, rcm in self.range_entries:
            key = f"range_{lo}_{hi}"
            out.update(scalar_metrics(result.to_flat_dict(f"{prefix}{key}/", class_names)))
            if rcm is not None and class_names:
                out.update(confusion_matrix_to_flat_dict(rcm, class_names, f"{prefix}{key}/"))
        return out

    def write_tensorboard(
        self,
        writer,
        class_names: List[str],
        epoch: int,
        prefix: str = "val/",
    ) -> None:
        """Write all metrics, per-class scalars, and CM figures to TensorBoard."""
        if writer is None:
            return
        import matplotlib.pyplot as plt

        _tb_write_scalars(writer, self.full.to_flat_dict("", class_names), prefix, epoch)
        for i, name in enumerate(class_names):
            for m in METRIC_NAMES:
                writer.add_scalar(
                    f"{prefix}class_{m}/{name}",
                    float(self.full.per_class[m][i]),
                    epoch,
                )

        _tb_write_scalars(writer, confusion_matrix_to_flat_dict(self.cm, class_names), prefix, epoch)
        fig = plot_confusion_matrix(self.cm, class_names)
        writer.add_figure(f"{prefix}confusion_matrix", fig, epoch)
        plt.close(fig)

        for lo, hi, result, rcm in self.range_entries:
            key = f"range_{lo}_{hi}"
            _tb_write_scalars(writer, result.to_flat_dict(f"range/{key}/", class_names), prefix, epoch)
            if rcm is not None:
                lbl = range_label(lo, hi)
                _tb_write_scalars(
                    writer,
                    confusion_matrix_to_flat_dict(rcm, class_names, f"range/{key}/"),
                    prefix,
                    epoch,
                )
                fig = plot_confusion_matrix(rcm, class_names, label=lbl)
                writer.add_figure(f"{prefix}confusion_matrix_{key}", fig, epoch)
                plt.close(fig)


def _tb_write_scalars(writer, metrics: dict, tag_prefix: str, step: int) -> None:
    """Write scalar entries of a metrics dict to TensorBoard."""
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            writer.add_scalar(f"{tag_prefix}{k}", float(v), step)


from mmengine.evaluator import BaseMetric  # noqa: E402


@METRICS.register_module()
class T4SegMetric(BaseMetric):
    """MMEngine BaseMetric for T4 3D segmentation.

    Named T4SegMetric to avoid conflict with mmdet3d's built-in SegMetric.
    Uses the same ``compute_metrics`` core as PTv3 so both frameworks
    report identical results.
    """

    default_prefix = "T4SegMetric"

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        class_names: Optional[List[str]] = None,
        distance_ranges: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or []
        self.distance_ranges = distance_ranges or []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            gt = _extract_array(data_sample, _GT_PATHS)
            pred = _extract_array(data_sample, _PRED_PATHS)
            if gt is None or pred is None or pred.size != gt.size:
                continue
            n = pred.size
            self.results.append(
                {
                    "pred": pred,
                    "gt": gt,
                    "coord": _extract_coord(data_sample, n),
                }
            )

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}
        nc, ign = self.num_classes, self.ignore_index
        cls_names = self.class_names if len(self.class_names) == nc else None

        full = CountsAccumulator(nc)
        ranges = CountsAccumulator(nc)
        cm = np.zeros((nc, nc), dtype=np.float64)
        range_cms: Dict[str, np.ndarray] = {}

        for r in results:
            pred, gt = r["pred"], r["gt"]
            i, u, t = intersection_union_target_np(pred, gt, nc, ign)
            full.add("full", i, u, t)
            cm += confusion_matrix_np(pred, gt, nc, ign)

            coord = r.get("coord")
            if self.distance_ranges and coord is not None and coord.size >= pred.size * 3:
                dist = bev_distance(coord.reshape(-1, 3)[: pred.size])
                for lo, hi in self.distance_ranges:
                    mask = (dist >= lo) & (dist < hi)
                    ranges.add_masked(f"range_{lo}_{hi}", pred, gt, mask, nc, ign)
                    if np.any(mask):
                        key = f"range_{lo}_{hi}"
                        if key not in range_cms:
                            range_cms[key] = np.zeros((nc, nc), dtype=np.float64)
                        range_cms[key] += confusion_matrix_np(pred[mask], gt[mask], nc, ign)

        fi, fu, ft = full.get("full")
        report = SegEvaluationReport.from_counts(
            fi,
            fu,
            ft,
            nc,
            cm,
            range_acc=ranges,
            range_cms=range_cms,
            distance_ranges=self.distance_ranges,
        )
        return report.to_flat_dict(cls_names, "T4SegMetric/")


_GT_PATHS = [
    ("gt_pts_seg", "pts_semantic_mask"),
    ("gt_pts_seg", "semantic_seg"),
]
_PRED_PATHS = [
    ("pred_pts_seg", "pts_semantic_mask"),
    ("pred_pts_seg", "seg_logits"),
]


def _to_numpy(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    if hasattr(v, "cpu"):
        v = v.cpu().numpy()
    arr = np.asarray(v)
    if arr.ndim > 1:
        arr = np.argmax(arr, axis=-1)
    return arr.ravel()


def _extract_array(sample, paths) -> Optional[np.ndarray]:
    """Try multiple attribute paths to extract a numpy array from a data sample."""
    for container_key, field_key in paths:
        container = getattr(sample, container_key, None)
        if container is None and isinstance(sample, dict):
            container = sample.get(container_key)
        if container is None:
            continue
        val = getattr(container, field_key, None)
        if val is None and isinstance(container, dict):
            val = container.get(field_key)
        if val is not None:
            return _to_numpy(val)
    return None


def _extract_coord(sample, n: int) -> Optional[np.ndarray]:
    for src in ("inputs",):
        container = getattr(sample, src, None) if not isinstance(sample, dict) else sample.get(src)
        if container is None:
            continue
        pts = container.get("points") if isinstance(container, dict) else getattr(container, "points", None)
        if pts is None:
            continue
        arr = pts.tensor[:, :3].cpu().numpy() if hasattr(pts, "tensor") else np.asarray(pts)[:, :3]
        return arr.reshape(-1, 3)[:n] if arr.size >= n * 3 else None
    return None


try:
    from mmdet3d.registry import METRICS as MMDET3D_METRICS

    MMDET3D_METRICS.register_module(module=T4SegMetric, force=True)
except Exception:
    pass
