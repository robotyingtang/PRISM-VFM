import torch
import torch.distributed as dist
import numpy as np


class SemsegMeter(object):

    def __init__(self, dataname, ignore_index=255):
        if dataname == "pascalcontext":
            n_classes = 20
            has_bg = True
        elif dataname == "nyud":
            n_classes = 40
            has_bg = False
        else:
            raise NotImplementedError

        self.ignore_index = ignore_index
        self.n_classes = n_classes + int(has_bg)

        # 关键修改1: 使用Tensor存储状态，而不是Python列表
        # 这使得我们可以使用高效的GPU all_reduce操作
        self.stats = torch.zeros(3, self.n_classes, dtype=torch.int64)  # 0: TP, 1: FP, 2: FN

    @property
    def tp(self):
        return self.stats[0]

    @property
    def fp(self):
        return self.stats[1]

    @property
    def fn(self):
        return self.stats[2]

    @torch.no_grad()
    def update(self, pred, gt):
        device = pred.device
        self.stats = self.stats.to(device)  # 确保状态Tensor在正确的GPU上

        pred, gt = pred.squeeze(), gt.squeeze()
        valid = gt != self.ignore_index

        for i_part in range(self.n_classes):
            tmp_gt = gt == i_part
            tmp_pred = pred == i_part
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid)
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid)
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid)

    def reset(self):
        self.stats.zero_()

    def get_score(self):
        # 关键修改2: 在计算前，同步所有GPU的统计数据
        # 打印本地（单个GPU）的统计量
        # print(f"[Rank {dist.get_rank()}] Before sync - TP sum: {self.tp.sum().item()}")

        synced_stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(synced_stats, op=dist.ReduceOp.SUM)

        # # 打印全局同步后的统计量
        # if dist.get_rank() == 0:
        #     print(f"[Rank 0] After sync - Global TP sum: {synced_stats[0].sum().item()}")

        # 从同步后的全局统计数据中提取 TP, FP, FN
        tp, fp, fn = synced_stats[0], synced_stats[1], synced_stats[2]

        jac = torch.zeros(self.n_classes, device=tp.device)
        for i_part in range(self.n_classes):
            denominator = tp[i_part] + fp[i_part] + fn[i_part]
            jac[i_part] = float(tp[i_part]) / max(float(denominator), 1e-8)

        # 只有 rank 0 需要返回最终结果，以避免重复日志
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        eval_result = {"mIoU": (torch.mean(jac).item() * 100)}
        return eval_result
