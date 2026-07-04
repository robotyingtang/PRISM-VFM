import torch
import torch.distributed as dist
import numpy as np


class DepthMeter(object):
    def __init__(self):
        self.max_depth = 80.0
        self.min_depth = 0.0
        # 关键修改1: 使用Tensor存储状态
        self.stats = torch.zeros(2, dtype=torch.float64)  # 0: total_rmses, 1: n_valid

    @property
    def total_rmses(self):
        return self.stats[0]

    @total_rmses.setter
    def total_rmses(self, value):
        self.stats[0] = value

    @property
    def n_valid(self):
        return self.stats[1]

    @n_valid.setter
    def n_valid(self, value):
        self.stats[1] = value

    @torch.no_grad()
    def update(self, pred, gt):
        self.stats = self.stats.to(pred.device)  # 确保状态在正确的GPU上
        pred, gt = pred.squeeze(), gt.squeeze()
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)

        self.n_valid += mask.float().sum()  # 移除 .item()

        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum()  # 移除 .item()

    def reset(self):
        self.stats.zero_()

    def get_score(self):
        # 关键修改2: 同步所有GPU的状态
        synced_stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(synced_stats, op=dist.ReduceOp.SUM)

        total_rmses, n_valid = synced_stats[0].item(), synced_stats[1].item()

        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        rmse = np.sqrt(total_rmses / n_valid) if n_valid > 0 else 0.0
        eval_result = {"RMSE": rmse}
        return eval_result
