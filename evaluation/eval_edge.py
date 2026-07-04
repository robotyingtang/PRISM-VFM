import torch
import torch.distributed as dist
import numpy as np


# 假设 BalancedBCELoss 在别处定义
from losses import BalancedBCELoss

class EdgeMeter(object):
    def __init__(self, dataname, ignore_index=255):
        if dataname == "pascalcontext":
            pos_weight = 0.95
        elif dataname == "nyud":
            pos_weight = 0.8
        else:
            raise NotImplementedError

        # loss_function 是 nn.Module, 让它保持原样
        self.loss_function = BalancedBCELoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        # 关键修改1: 使用Tensor存储状态
        self.stats = torch.zeros(2, dtype=torch.float64)  # 0: total_loss, 1: total_n

    @property
    def loss(self):
        return self.stats[0]

    @loss.setter
    def loss(self, value):
        self.stats[0] = value

    @property
    def n(self):
        return self.stats[1]

    @n.setter
    def n(self, value):
        self.stats[1] = value

    @torch.no_grad()
    def update(self, pred, gt):
        self.stats = self.stats.to(pred.device)
        pred, gt = pred.squeeze(), gt.squeeze()
        valid_mask = gt != self.ignore_index

        # 将loss_function也移动到正确的设备
        self.loss_function = self.loss_function.to(pred.device)

        loss = self.loss_function(pred, gt)  # pred需要保留在mask前以匹配gt形状

        pred = pred[valid_mask]
        gt = gt[valid_mask]

        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss.item()  # loss计算可能涉及整个batch，这里用item()是安全的

    def reset(self):
        self.stats.zero_()

    def get_score(self):
        # 关键修改2: 同步所有GPU的状态
        synced_stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(synced_stats, op=dist.ReduceOp.SUM)

        total_loss, total_n = synced_stats[0].item(), synced_stats[1].item()

        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        avg_loss = total_loss / total_n if total_n > 0 else 0.0
        eval_dict = {"loss": avg_loss}
        return eval_dict
