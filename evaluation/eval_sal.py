import torch
import torch.distributed as dist
import numpy as np


class SaliencyMeter(object):
    def __init__(self, ignore_index=255, threshold_step=0.05, beta=0.3):
        self.ignore_index = ignore_index
        self.beta = beta
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)

        # 关键修改1: 使用Tensor存储状态
        # stats: [4, num_thresholds]
        # 0: true_positives, 1: predicted_positives, 2: actual_positives
        self.stats = torch.zeros(3, len(self.thresholds), dtype=torch.float64)
        # ious 仍然可以是列表，因为它只在 rank 0 上有意义
        self.ious = []

    @property
    def true_positives(self):
        return self.stats[0]

    @property
    def predicted_positives(self):
        return self.stats[1]

    @property
    def actual_positives(self):
        return self.stats[2]

    @torch.no_grad()
    def update(self, preds, targets):
        self.stats = self.stats.to(preds.device)
        self.thresholds = self.thresholds.to(preds.device)
        preds = preds.float() / 255.0

        if targets.shape[1] == 1: targets = targets.squeeze(1)
        assert preds.shape == targets.shape

        for i in range(preds.size(0)):
            pred, target = preds[i], targets[i]
            valid_mask = target != self.ignore_index
            iou_per_image = np.zeros(len(self.thresholds))

            for idx, thresh in enumerate(self.thresholds):
                f_pred = (pred >= thresh).long()
                f_target = target.long()
                f_pred_masked = torch.masked_select(f_pred, valid_mask)
                f_target_masked = torch.masked_select(f_target, valid_mask)

                self.true_positives[idx] += torch.sum(f_pred_masked * f_target_masked)
                self.predicted_positives[idx] += torch.sum(f_pred_masked)
                self.actual_positives[idx] += torch.sum(f_target_masked)

                # IoU 只在 rank 0 上计算和收集
                if not dist.is_initialized() or dist.get_rank() == 0:
                    iou_per_image[idx] = (torch.sum(f_pred_masked & f_target_masked).item() /
                                          torch.sum(f_pred_masked | f_target_masked).item())

            if not dist.is_initialized() or dist.get_rank() == 0:
                self.ious.append(iou_per_image)

    def get_score(self):
        # 关键修改2: 同步所有GPU的状态
        synced_stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(synced_stats, op=dist.ReduceOp.SUM)

        # 只有 rank 0 需要返回最终结果
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        tp, pp, ap = synced_stats[0], synced_stats[1], synced_stats[2]

        precision = tp / (pp + 1e-8)
        recall = tp / (ap + 1e-8)
        fscore = (1 + self.beta) * precision * recall / (self.beta * precision + recall + 1e-8)
        fscore[fscore != fscore] = 0

        # # mIoUs 的计算逻辑现在只在 rank 0 上有意义
        # mIoUs = np.mean(np.array(self.ious), axis=0) if self.ious else np.zeros(len(self.thresholds))

        # --- 修正 mIoU 的计算 ---
        # IoU = TP / (TP + FP + FN) = TP / (PP + AP - TP)
        iou_per_threshold = tp / (pp + ap - tp + 1e-8)
        iou_per_threshold[iou_per_threshold != iou_per_threshold] = 0  # handle NaNs
        mIoU = iou_per_threshold.max().item()
        # -------------------------

        eval_result = {"maxF": (fscore.max().item() * 100), "mIoU": (mIoU * 100)}
        return eval_result
