"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from autoware_ml.segmentation3d.datasets.utils import class_mapping_to_names
from autoware_ml.segmentation3d.evaluation import (
    CountsAccumulator,
    SegEvaluationReport,
)

from .builder import HOOKS
from .default import HookBase


def _bev_distance(coord: torch.Tensor) -> torch.Tensor:
    """BEV radius from ego: sqrt(x^2 + y^2)."""
    return torch.sqrt(coord[:, 0].float() ** 2 + coord[:, 1].float() ** 2)


def _confusion_matrix_gpu(pred, segment, num_classes, ignore_index):
    """Compute (num_classes, num_classes) confusion matrix on GPU."""
    valid = (segment != ignore_index) & (segment >= 0) & (segment < num_classes) & (pred >= 0) & (pred < num_classes)
    indices = (segment[valid] * num_classes + pred[valid]).long()
    return torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        cfg = self.trainer.cfg
        num_classes = cfg.data.num_classes
        ignore_index = cfg.data.ignore_index
        metric_options = getattr(cfg, "metric_options", None) or {}
        distance_ranges = metric_options.get("distance_ranges", [])

        range_acc = CountsAccumulator(num_classes)
        cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")
        range_cm_acc: dict = {}

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                num_classes,
                ignore_index,
            )
            cm_batch = _confusion_matrix_gpu(pred, segment, num_classes, ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                dist.all_reduce(cm_batch)
            cm_total += cm_batch

            self.trainer.storage.put_scalar("val_intersection", intersection.cpu().numpy())
            self.trainer.storage.put_scalar("val_union", union.cpu().numpy())
            self.trainer.storage.put_scalar("val_target", target.cpu().numpy())
            self.trainer.storage.put_scalar("val_loss", loss.item())

            if distance_ranges and "coord" in input_dict:
                coord = input_dict["coord"]
                if coord.numel() >= pred.numel() * 3:
                    dist_bev = _bev_distance(coord.view(-1, 3)[: pred.numel()])
                    for lo, hi in distance_ranges:
                        mask = (dist_bev >= lo) & (dist_bev < hi)
                        if not mask.any():
                            continue
                        i_r, u_r, t_r = intersection_and_union_gpu(
                            pred[mask], segment[mask], num_classes, ignore_index
                        )
                        cm_r = _confusion_matrix_gpu(pred[mask], segment[mask], num_classes, ignore_index)
                        if comm.get_world_size() > 1:
                            dist.all_reduce(i_r), dist.all_reduce(u_r), dist.all_reduce(t_r)
                            dist.all_reduce(cm_r)
                        range_acc.add(f"range_{lo}_{hi}", i_r.cpu().numpy(), u_r.cpu().numpy(), t_r.cpu().numpy())
                        key = f"range_{lo}_{hi}"
                        if key not in range_cm_acc:
                            range_cm_acc[key] = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")
                        range_cm_acc[key] += cm_r

            info = "Test: [{}/{}] ".format(i + 1, len(self.trainer.val_loader))
            if "origin_coord" in input_dict:
                info = "Interp. " + info
            self.trainer.logger.info(info + "Loss {:.4f} ".format(loss.item()))

        cm = cm_total.cpu().numpy().astype(np.float64)
        range_cms = {k: v.cpu().numpy().astype(np.float64) for k, v in range_cm_acc.items()}
        self._log_final_metrics(cfg, num_classes, ignore_index, distance_ranges, range_acc, cm, range_cms)

    def _log_final_metrics(self, cfg, num_classes, ignore_index, distance_ranges, range_acc, cm, range_cms):
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total

        mapped_class_names = class_mapping_to_names(cfg.class_mapping, ignore_index)
        assert len(mapped_class_names) == num_classes

        report = SegEvaluationReport.from_counts(
            intersection,
            union,
            target,
            num_classes,
            cm,
            range_acc=range_acc,
            range_cms=range_cms,
            distance_ranges=distance_ranges,
        )
        report.log(self.trainer.logger, mapped_class_names, label="Val")

        writer = self.trainer.writer
        epoch = self.trainer.epoch + 1
        if writer is not None:
            writer.add_scalar("val/loss", loss_avg, epoch)
        report.write_tensorboard(writer, mapped_class_names, epoch)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = report.full.class_avg["iou"]
        self.trainer.comm_info["current_metric_name"] = "iou_class"

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format("iou_class", self.trainer.best_metric_value))
