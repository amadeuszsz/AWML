"""ONNX export script for PT-v3m2 (Sonata) + linear segmentation head.

The export strategy mirrors the m1-base deploy pattern:

1. Serialization (z-order encoding) is pre-computed *outside* the traced graph
   and passed as extra inputs (``serialized_code``, ``serialized_depth``).
2. Inside the graph, ``WrappedModel.forward`` recomputes ``serialized_order``
   and ``serialized_inverse`` from ``serialized_code`` using the custom
   ``argsort`` op (``autoware::Argsort``) so that these remain dynamic.
3. ``GridPooling`` modules that appear later in the encoder *do* call
   ``serialization()`` internally (unlike m1's ``SerializedPooling``).  The
   underlying ``encode`` path has been made ONNX / TRT-safe with arithmetic
   ops and the ``pow2`` LUT.
"""

import numpy as np
import SparseConvolution  # noqa: F401 – registers ONNX symbolic ops
import spconv.pytorch as spconv
import torch
from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.train import TRAINERS
from models.scatter.functional import argsort
from models.utils.structure import Point, bit_length_tensor
from torch.nn import functional as F


class WrappedModel(torch.nn.Module):
    """Thin wrapper that prepares inputs for ``export_forward``."""

    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model.cuda()
        self.model.backbone.forward = self.model.backbone.export_forward

        point_cloud_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32).cuda()
        voxel_size = cfg.grid_size
        voxel_size = torch.tensor([voxel_size] * 3, dtype=torch.float32).cuda()

        self.sparse_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.sparse_shape = torch.round(self.sparse_shape).long().cuda()

    def forward(
        self,
        grid_coord,
        feat,
        serialized_depth,
        serialized_code,
    ):
        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)

        # Recompute order/inverse from code inside the traced graph so that
        # they stay dynamic (matching m1-base deploy pattern).
        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(0, serialized_code.shape[1], device=serialized_order.device).repeat(
                serialized_code.shape[0], 1
            ),
        )

        input_dict = {
            "coord": feat[:, :3],
            "grid_coord": grid_coord,
            "offset": shape[:1],
            "feat": feat,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "sparse_shape": self.sparse_shape,
        }

        output = self.model(input_dict)

        pred_logits = output["seg_logits"]  # (n, k)
        pred_probs = F.softmax(pred_logits, -1)
        pred_label = pred_probs.argmax(-1)

        return pred_label, pred_probs


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker = 1
    cfg.num_worker_per_gpu = 1

    # Force export-compatible settings
    cfg.model.backbone.shuffle_orders = False
    cfg.model.backbone.order = ["z", "z-trans"]
    cfg.model.backbone.export_mode = True

    runner = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    runner.before_train()

    # --- Reload trained checkpoint directly ---
    # The config's ``CheckpointLoader`` may apply keyword replacement (e.g.
    # for Concerto pretrain loading) that would mangle keys of a full model
    # checkpoint (double "backbone." prefix).  Detect and bypass if needed.
    weight_path = cfg.weight
    if weight_path:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        has_backbone = any(k.startswith("backbone.") for k in state_dict)
        has_seg_head = any(k.startswith("seg_head") for k in state_dict)
        if has_backbone and has_seg_head:
            info = runner.model.load_state_dict(state_dict, strict=False)
            print(f"[export] Direct load — missing: {info.missing_keys}, " f"unexpected: {info.unexpected_keys}")

    model = WrappedModel(runner.model, cfg)
    model.eval()

    # Grab one sample for tracing
    runner.val_loader.prefetch_factor = 1
    data_dict = next(iter(runner.val_loader))

    input_dict = data_dict
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

    with torch.no_grad():
        # Pre-compute serialization outside the traced graph (static depth)
        depth = bit_length_tensor(
            torch.tensor([(max(cfg.point_cloud_range) - min(cfg.point_cloud_range)) / cfg.grid_size])
        ).cuda()
        point = Point(input_dict)
        point.serialization(
            order=model.model.backbone.order,
            shuffle_orders=model.model.backbone.shuffle_orders,
            depth=depth,
        )

        input_dict["serialized_depth"] = point["serialized_depth"]
        input_dict["serialized_code"] = point["serialized_code"]
        input_dict.pop("segment", None)
        input_dict.pop("offset", None)
        input_dict.pop("coord", None)

        # Sanity-check: run the model once before export
        pred_labels, pred_probs = model(**input_dict)

        import os

        save_dir = cfg.save_path if hasattr(cfg, "save_path") else "."
        os.makedirs(save_dir, exist_ok=True)
        npz_path = os.path.join(save_dir, "ptv3_sample.npz")
        onnx_path = os.path.join(save_dir, "ptv3.onnx")
        np.savez_compressed(
            npz_path,
            pred=pred_labels.cpu().numpy(),
            feat=input_dict["feat"].cpu().numpy(),
        )

        # ONNX export
        input_names = ["grid_coord", "feat", "serialized_depth", "serialized_code"]
        output_names = ["pred_labels", "pred_probs"]
        dynamic_axes = {
            "grid_coord": {0: "voxels_num"},
            "feat": {0: "voxels_num"},
            "serialized_code": {1: "voxels_num"},
        }
        torch.onnx.export(
            model,
            input_dict,
            onnx_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=False,
            verbose=False,
            do_constant_folding=False,
        )

    print(f"Exported to {onnx_path} successfully.")


if __name__ == "__main__":
    main()
