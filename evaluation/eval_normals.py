import torch
import torch.distributed as dist
import numpy as np

def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p="fro", dim=dim, keepdim=True)
    zero_mask = norm == 0
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


class NormalsMeter(object):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        # 关键修改1: 使用Tensor存储状态
        self.stats = torch.zeros(2, dtype=torch.float64)  # 0: sum_deg_diff, 1: total

    @property
    def sum_deg_diff(self):
        return self.stats[0]

    @sum_deg_diff.setter
    def sum_deg_diff(self, value):
        self.stats[0] = value

    @property
    def total(self):
        return self.stats[1]

    @total.setter
    def total(self, value):
        self.stats[1] = value

    @torch.no_grad()
    def update(self, pred, gt):
        self.stats = self.stats.to(pred.device)
        pred = pred.permute(0, 3, 1, 2)
        pred = 2 * pred / 255 - 1
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = normalize_tensor(pred, dim=1)
        gt = normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff)  # 移除 .item()
        self.total += deg_diff.numel()

    def get_score(self):
        # 关键修改2: 同步所有GPU的状态
        synced_stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(synced_stats, op=dist.ReduceOp.SUM)

        sum_deg_diff, total = synced_stats[0].item(), synced_stats[1].item()

        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        mErr = sum_deg_diff / total if total > 0 else 0.0
        eval_result = {"mErr": mErr}
        return eval_result
