#!/usr/bin/env python3
"""Split T4 dataset scenes into train/val/test by spatial distribution of ego trajectories.

Strategies:
  clusters  – one K-means cluster per non-zero split; split centroids are maximally varied.
  uniform   – stratified by cluster, then refined so split centroids are as close as possible.

Number of K-means clusters equals the number of non-zero split ratios unless overridden.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import numpy as np
import yaml
from mmengine.logging import print_log
from sklearn.cluster import KMeans
from t4_devkit import Tier4
from t4_devkit.schema import Sample, SampleData

_SPLIT_NAMES = ("train", "val", "test")


def _get_lidar_token(sample: Sample) -> str | None:
    data = sample.data
    if "LIDAR_TOP" in data:
        return data["LIDAR_TOP"]
    if "LIDAR_CONCAT" in data:
        return data["LIDAR_CONCAT"]
    return None


def get_scene_trajectory_xy(scene_root: str) -> tuple[list[list[float]], int] | None:
    """Return 2D ego trajectory ``[(x, y), ...]`` from ``ego_pose.translation`` for every
    lidar frame in the scene, or *None* if the scene cannot be loaded."""
    try:
        t4 = Tier4(data_root=scene_root, verbose=False)
    except Exception:
        return None
    trajectory: list[list[float]] = []
    for sample in t4.sample:
        lidar_token = _get_lidar_token(sample)
        if lidar_token is None:
            continue
        sd_record: SampleData = t4.get("sample_data", lidar_token)
        pose = t4.get("ego_pose", sd_record.ego_pose_token)
        t = np.asarray(pose.translation, dtype=np.float64)
        trajectory.append([float(t[0]), float(t[1])])
    if not trajectory:
        return None
    return trajectory, len(trajectory)


def compute_scene_descriptor(trajectory: list[list[float]]) -> dict[str, Any]:
    """Centroid and bounding box of a 2D trajectory."""
    arr = np.asarray(trajectory, dtype=np.float64)
    centroid = arr.mean(axis=0).tolist()
    mn = arr.min(axis=0).tolist()
    mx = arr.max(axis=0).tolist()
    return {
        "centroid_xy": centroid,
        "bbox_xy": {"min_x": mn[0], "min_y": mn[1], "max_x": mx[0], "max_y": mx[1]},
        "num_points": len(trajectory),
    }


def discover_scenes(input_path: str) -> list[tuple[str, str]]:
    """Return ``[(scene_id, scene_root), ...]`` for every versioned scene directory.

    ``scene_id`` has format ``{uuid}/{version}``.  The ``info/`` subdirectory is ignored.
    """
    version_re = re.compile(r"^\d+$")
    if not os.path.isdir(input_path):
        return []
    results: list[tuple[str, str]] = []
    for name in sorted(os.listdir(input_path)):
        if name == "info":
            continue
        uuid_dir = os.path.join(input_path, name)
        if not os.path.isdir(uuid_dir):
            continue
        version_dirs = sorted(
            (d for d in os.listdir(uuid_dir) if version_re.match(d)),
            key=int,
        )
        for ver in version_dirs:
            scene_root = os.path.join(uuid_dir, ver)
            if os.path.isdir(scene_root):
                results.append((f"{name}/{ver}", scene_root))
    return results


def _centroid_stats(centroids: np.ndarray, indices: list[int]) -> dict[str, float]:
    """Mean / std / min / max of (x, y) for the given centroid indices.

    Raises ``ValueError`` when *indices* is empty — returning zeros would corrupt
    downstream calculations in global coordinate systems.
    """
    if not indices:
        raise ValueError("Cannot compute centroid stats for an empty index list.")
    pts = centroids[indices]
    mean = pts.mean(axis=0)
    std = pts.std(axis=0) if len(pts) > 1 else np.zeros(2)
    return {
        "mean_x": float(mean[0]),
        "mean_y": float(mean[1]),
        "std_x": float(std[0]),
        "std_y": float(std[1]),
        "min_x": float(pts[:, 0].min()),
        "min_y": float(pts[:, 1].min()),
        "max_x": float(pts[:, 0].max()),
        "max_y": float(pts[:, 1].max()),
    }


def _split_centroid_spread(
    centroids: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> float:
    """Sum of pairwise squared distances between split mean-centroids.

    Raises ``ValueError`` when any split list is empty.
    """
    for name, idx in zip(_SPLIT_NAMES, [train_idx, val_idx, test_idx]):
        if not idx:
            raise ValueError(f"Cannot compute centroid spread: {name} split is empty.")
    means = [centroids[idx].mean(axis=0) for idx in (train_idx, val_idx, test_idx)]
    spread = sum(float(np.sum((means[i] - means[j]) ** 2)) for i in range(3) for j in range(i + 1, 3))
    return spread


def _n_clusters_from_ratios(ratios: tuple[float, float, float], eps: float = 1e-9) -> int:
    """Number of non-zero split ratios (e.g. ``(0.5, 0.0, 0.5)`` → 2)."""
    return sum(1 for r in ratios if r > eps)


def strategy_clusters(
    scene_ids: list[str],
    centroids: np.ndarray,
    ratios: tuple[float, float, float],
    n_clusters: int,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[int]]:
    """One K-means cluster per non-zero split → maximally varied split centroids."""
    k = max(1, min(n_clusters, len(scene_ids)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(centroids)
    cluster_to_scenes: dict[int, list[int]] = {c: [] for c in range(k)}
    for i, c in enumerate(labels):
        cluster_to_scenes[c].append(i)

    split_map = {i: name for i, name in enumerate(_SPLIT_NAMES) if ratios[i] > 1e-9}
    buckets: dict[str, set[int]] = {name: set() for name in _SPLIT_NAMES}
    for c in range(k):
        target = split_map.get(c, _SPLIT_NAMES[0])
        buckets[target].update(cluster_to_scenes[c])

    return (
        [scene_ids[i] for i in sorted(buckets["train"])],
        [scene_ids[i] for i in sorted(buckets["val"])],
        [scene_ids[i] for i in sorted(buckets["test"])],
        labels.tolist(),
    )


def strategy_uniform(
    scene_ids: list[str],
    centroids: np.ndarray,
    ratios: tuple[float, float, float],
    n_clusters: int,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Stratified split by cluster, then greedy swap-refinement to minimize centroid spread."""
    rng = np.random.default_rng(seed)
    k = max(1, min(n_clusters, len(scene_ids)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(centroids)

    cluster_to_scenes: dict[int, list[int]] = {c: [] for c in range(k)}
    for i, c in enumerate(labels):
        cluster_to_scenes[c].append(i)

    split_lists: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for scene_idxs in cluster_to_scenes.values():
        perm = rng.permutation(len(scene_idxs))
        n = len(scene_idxs)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        for p in range(n):
            idx = scene_idxs[perm[p]]
            if p < n_train:
                split_lists["train"].append(idx)
            elif p < n_train + n_val:
                split_lists["val"].append(idx)
            else:
                split_lists["test"].append(idx)

    tr, va, te = split_lists["train"], split_lists["val"], split_lists["test"]
    if tr and va and te:
        _refine_uniform(centroids, tr, va, te, max_passes=30)

    return (
        [scene_ids[i] for i in sorted(tr)],
        [scene_ids[i] for i in sorted(va)],
        [scene_ids[i] for i in sorted(te)],
        labels.tolist(),
    )


def _refine_uniform(
    centroids: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    max_passes: int = 30,
) -> None:
    """Greedy pair-swap refinement (in-place) to minimize ``_split_centroid_spread``."""
    spread = _split_centroid_spread(centroids, train_idx, val_idx, test_idx)
    for _ in range(max_passes):
        improved = False
        for list_a, list_b in [
            (train_idx, val_idx),
            (train_idx, test_idx),
            (val_idx, test_idx),
        ]:
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    list_a[i], list_b[j] = list_b[j], list_a[i]
                    new_spread = _split_centroid_spread(centroids, train_idx, val_idx, test_idx)
                    if new_spread < spread:
                        spread = new_spread
                        improved = True
                        break
                    list_a[i], list_b[j] = list_b[j], list_a[i]
                if improved:
                    break
            if improved:
                break
        if not improved:
            break


def _log_table(title: str, body: str) -> None:
    print_log(title, logger="current")
    for line in body.splitlines():
        print_log(line, logger="current")


def _fmt_kv(d: dict[str, Any], key_width: int = 24) -> str:
    lines: list[str] = []
    for k, v in d.items():
        val = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
        lines.append(f"  {k:<{key_width}}  {val}")
    return "\n".join(lines)


def _fmt_skipped(skipped: list[dict[str, str]]) -> str:
    if not skipped:
        return "  (none)"
    cw = (50, 32)
    lines = [
        f"  {'scene_id':<{cw[0]}}  {'reason':<{cw[1]}}",
        "  " + "-" * (sum(cw) + 2),
    ]
    for s in skipped:
        lines.append(f"  {s['scene_id']:<{cw[0]}}  {s['reason']:<{cw[1]}}")
    return "\n".join(lines)


_STAT_COLS = ("mean_x", "mean_y", "std_x", "std_y", "min_x", "min_y", "max_x", "max_y")
_STAT_WIDTHS = (12, 12, 10, 10, 12, 12, 12, 12)


def _fmt_cluster_summary(cs: dict[str, Any]) -> str:
    if not cs:
        return "  (none)"
    cols = ("cluster", "count", "train", "val", "test") + _STAT_COLS
    widths = (8, 6, 6, 4, 5) + _STAT_WIDTHS
    header = "  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  " + "-" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [header, sep]
    for cid, row in sorted(cs.items(), key=lambda x: int(x[0])):
        cells = [str(cid), str(row["count"]), str(row["train"]), str(row["val"]), str(row["test"])]
        cells += [f"{row[c]:.2f}" for c in _STAT_COLS]
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def _fmt_split_summary(ss: dict[str, Any]) -> str:
    if not ss:
        return "  (none)"
    cols = ("split", "count") + _STAT_COLS
    widths = (8, 6) + _STAT_WIDTHS
    header = "  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  " + "-" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [header, sep]
    for name in _SPLIT_NAMES:
        row = ss.get(name)
        if row is None:
            continue
        cells = [name, str(row["count"])] + [f"{row[c]:.2f}" for c in _STAT_COLS]
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split T4 dataset scenes by spatial distribution of ego trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="Path to the scenes database directory.")
    p.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        required=True,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train / val / test ratios (must sum to 1.0).",
    )
    p.add_argument(
        "--strategy",
        choices=["clusters", "uniform"],
        required=True,
        help="clusters: separate regions per split. uniform: similar distribution per split.",
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Override K-means cluster count (default: number of non-zero ratios).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("-o", "--out", required=True, help="Output YAML path (must end with .yaml).")
    p.add_argument("--version", type=int, default=1, help="'version' field in output YAML.")
    p.add_argument("--dataset_version", required=True, help="'dataset_version' field in output YAML.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = tuple(args.split_ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratios}")

    yaml_path: str = args.out
    if not yaml_path.endswith(".yaml"):
        raise ValueError(f"--out must end with .yaml, got {yaml_path!r}")
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)) or ".", exist_ok=True)

    # --- discover & load scenes ---
    print_log(f"Discovering scenes under {args.input!r} ...", logger="current")
    scenes = discover_scenes(args.input)
    if not scenes:
        raise SystemExit(f"No scenes found under {args.input!r} (excluding 'info').")
    print_log(f"Found {len(scenes)} scene(s). Loading trajectories ...", logger="current")

    scene_ids: list[str] = []
    descriptors: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for idx, (scene_id, scene_root) in enumerate(scenes):
        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(scenes) - 1:
            print_log(f"  Loading scene {idx + 1}/{len(scenes)}: {scene_id}", logger="current")
        result = get_scene_trajectory_xy(scene_root)
        if result is None:
            skipped.append({"scene_id": scene_id, "reason": "load_failed_or_no_lidar"})
            continue
        trajectory, _ = result
        scene_ids.append(scene_id)
        descriptors.append(compute_scene_descriptor(trajectory))

    if not scene_ids:
        raise SystemExit("No scenes could be loaded (all failed or no lidar).")
    print_log(f"Loaded {len(scene_ids)} scene(s), skipped {len(skipped)}.", logger="current")

    # --- cluster & split ---
    centroids = np.array([d["centroid_xy"] for d in descriptors], dtype=np.float64)
    n_clusters = args.n_clusters if args.n_clusters is not None else _n_clusters_from_ratios(ratios)
    source = "user-specified" if args.n_clusters is not None else "from non-zero split ratios"
    print_log(f"n_clusters={n_clusters} ({source}), strategy={args.strategy}.", logger="current")

    if args.strategy == "clusters":
        print_log("Assigning clusters to splits (one cluster per split) ...", logger="current")
        train_ids, val_ids, test_ids, cluster_labels = strategy_clusters(
            scene_ids,
            centroids,
            ratios,
            n_clusters,
            args.seed,
        )
    else:
        print_log("Stratifying by cluster and refining centroid spread ...", logger="current")
        train_ids, val_ids, test_ids, cluster_labels = strategy_uniform(
            scene_ids,
            centroids,
            ratios,
            n_clusters,
            args.seed,
        )

    print_log(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}.", logger="current")

    # --- validate non-zero-ratio splits are not empty ---
    eps = 1e-9
    for i, (name, ids) in enumerate(zip(_SPLIT_NAMES, [train_ids, val_ids, test_ids])):
        if ratios[i] > eps and len(ids) == 0:
            raise ValueError(
                f"{name} split is empty but ratio={ratios[i]:.3f}. " "Increase dataset size or adjust split_ratios."
            )

    # --- build index look-ups (for summary tables) ---
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    split_indices: dict[str, list[int]] = {
        "train": [i for i, s in enumerate(scene_ids) if s in train_set],
        "val": [i for i, s in enumerate(scene_ids) if s in val_set],
        "test": [i for i, s in enumerate(scene_ids) if s in test_set],
    }

    cluster_summary: dict[str, dict[str, Any]] = {}
    for c in set(cluster_labels):
        idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
        cluster_summary[str(c)] = {
            "count": len(idxs),
            "train": sum(1 for i in idxs if scene_ids[i] in train_set),
            "val": sum(1 for i in idxs if scene_ids[i] in val_set),
            "test": sum(1 for i in idxs if scene_ids[i] in test_set),
            **_centroid_stats(centroids, idxs),
        }

    split_summary: dict[str, dict[str, Any]] = {}
    for name in _SPLIT_NAMES:
        idxs = split_indices[name]
        if idxs:
            split_summary[name] = {"count": len(idxs), **_centroid_stats(centroids, idxs)}

    # --- write YAML ---
    yaml_data = {
        "version": args.version,
        "dataset_version": args.dataset_version,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    print_log(f"Writing YAML to {yaml_path} ...", logger="current")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # --- structured log ---
    config_table = {
        "input": args.input,
        "dataset_version": args.dataset_version,
        "split_ratios": list(ratios),
        "strategy": args.strategy,
        "n_clusters": n_clusters,
        "seed": args.seed,
        "version": args.version,
    }
    summary_table = {
        "total_scenes_found": len(scenes),
        "total_scenes_used": len(scene_ids),
        "skipped_count": len(skipped),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
    }

    print_log("========== Split result ==========", logger="current")
    _log_table("--- config ---", _fmt_kv(config_table))
    _log_table("--- summary ---", _fmt_kv(summary_table))
    _log_table("--- skipped ---", _fmt_skipped(skipped))
    _log_table("--- cluster_summary ---", _fmt_cluster_summary(cluster_summary))
    _log_table("--- split_summary ---", _fmt_split_summary(split_summary))
    print_log("==================================", logger="current")
    print_log(
        f"Done. Wrote {yaml_path} | "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} "
        f"(skipped {len(skipped)}).",
        logger="current",
    )


if __name__ == "__main__":
    main()
