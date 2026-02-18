# Dataset Config Tools

## simple_split_t4dataset.py

Split T4 dataset scenes into **train / val / test** based on the spatial distribution of ego-vehicle trajectories (2D x, y from `ego_pose.translation`).

### Strategies

| Strategy   | Behavior                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clusters` | Each split receives a separate K-means cluster — split centroids are **maximally varied** (good for testing cross-region generalisation).                      |
| `uniform`  | Scenes are stratified by cluster and then refined via greedy swaps so that split centroids are **as close as possible** (good for in-distribution evaluation). |

The number of K-means clusters defaults to the number of non-zero split ratios (e.g. `0.7 0.3 0.0` → 2 clusters) but can be overridden with `--n_clusters`.

### Usage

```bash
python tools/dataset_config/simple_split_t4dataset.py \
    --input data/t4dataset/db_j6gen2_semaseg_v1 \
    --split_ratios 0.7 0.2 0.1 \
    --strategy uniform \
    --dataset_version db_j6gen2_semaseg_v1 \
    --out autoware_ml/configs/t4dataset/db_j6gen2_semaseg_v1.yaml
```

### Arguments

| Argument            | Required | Default | Description                                     |
| ------------------- | -------- | ------- | ----------------------------------------------- |
| `-i`, `--input`     | yes      | —       | Path to the scenes database directory.          |
| `--split_ratios`    | yes      | —       | Three floats (train, val, test) summing to 1.0. |
| `--strategy`        | yes      | —       | `clusters` or `uniform`.                        |
| `-o`, `--out`       | yes      | —       | Output YAML path (must end with `.yaml`).       |
| `--dataset_version` | yes      | —       | Value for `dataset_version` in the output YAML. |
| `--version`         | no       | 1       | Value for `version` in the output YAML.         |
| `--n_clusters`      | no       | auto    | Override K-means cluster count.                 |
| `--seed`            | no       | 42      | Random seed for reproducibility.                |

### Output

- **YAML** file compatible with `autoware_ml/configs/t4dataset/*.yaml` (contains `version`, `dataset_version`, `train`, `val`, `test` lists).
- **Structured log** printed to stdout with config, summary, skipped scenes, cluster statistics, and per-split statistics.
