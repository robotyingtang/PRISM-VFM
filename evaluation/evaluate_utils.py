from utils import get_output

from .save_img import save_img


class PerformanceMeter(object):
    """
    A general performance meter which shows performance across one or more tasks
    """

    def __init__(self, dataname, tasks):
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(dataname, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score()

        return eval_dict


def get_single_task_meter(dataname, task):
    """
    Retrieve a meter to measure the single-task performance
    """

    if task == "semseg":
        from .eval_semseg import SemsegMeter

        return SemsegMeter(dataname)

    elif task == "human_parts":
        from .eval_human_parts import HumanPartsMeter

        return HumanPartsMeter()

    elif task == "normals":
        from .eval_normals import NormalsMeter

        return NormalsMeter()

    elif task == "sal":
        from .eval_sal import SaliencyMeter

        return SaliencyMeter()

    elif task == "depth":
        from .eval_depth import DepthMeter

        return DepthMeter()

    elif (
        task == "edge"
    ):  # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from .eval_edge import EdgeMeter

        return EdgeMeter(dataname)

    else:
        raise NotImplementedError


def predict(dataname, meta, outputs, task, pred_dir):
    """
    Get predictions and save predicted images
    :param str dataname: Dataset name
    :param dict meta: Metadata from the dataset, containing image names and sizes
    :param dict outputs: Model outputs
    :param str task: Task name
    :param str pred_dir: Directory to save the predictions
    """

    output_task = get_output(outputs[task], task)
    preds = []
    for i in range(output_task.size(0)):
        # Cut image borders (padding area)
        pred = output_task[i]  # H, W or H, W, C
        ori_dim = (int(meta["size"][i][0]), int(meta["size"][i][1]))
        curr_dim = tuple(pred.shape[:2])

        if ori_dim != curr_dim:
            # Height and width of border
            delta_h = max(curr_dim[0] - ori_dim[0], 0)
            delta_w = max(curr_dim[1] - ori_dim[1], 0)

            # Location of original image
            loc_h = [delta_h // 2, (delta_h // 2) + ori_dim[0]]
            loc_w = [delta_w // 2, (delta_w // 2) + ori_dim[1]]

            pred = pred[loc_h[0] : loc_h[1], loc_w[0] : loc_w[1]]

        pred = pred.cpu().numpy()
        preds.append(pred)

    save_img(dataname, meta["file_name"], preds, task, pred_dir)



import os
import numpy as np
import torch
from PIL import Image


def save_input_img(meta, images, pred_dir, mean=None, std=None):
    """
    Save input images to pred_dir/img

    :param dict meta: batch meta, containing file_name and size
    :param torch.Tensor images: [B, C, H, W]
    :param str pred_dir: root prediction dir
    :param list mean: optional normalization mean, e.g. [0.485, 0.456, 0.406]
    :param list std: optional normalization std, e.g. [0.229, 0.224, 0.225]
    """
    img_dir = os.path.join(pred_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    for i in range(images.size(0)):
        img = images[i].detach().cpu().float()  # [C, H, W]

        # 如果 dataloader 对图像做过 normalize，这里反归一化
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            img = img * std_t + mean_t

        # CHW -> HWC
        img = img.permute(1, 2, 0).numpy()

        # 按 meta["size"] 裁掉 padding，和 predict() 里保持一致
        ori_dim = (int(meta["size"][i][0]), int(meta["size"][i][1]))  # (H, W)
        curr_dim = tuple(img.shape[:2])

        if ori_dim != curr_dim:
            delta_h = max(curr_dim[0] - ori_dim[0], 0)
            delta_w = max(curr_dim[1] - ori_dim[1], 0)

            loc_h = [delta_h // 2, (delta_h // 2) + ori_dim[0]]
            loc_w = [delta_w // 2, (delta_w // 2) + ori_dim[1]]

            img = img[loc_h[0]:loc_h[1], loc_w[0]:loc_w[1]]

        # 转成 uint8 保存
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        file_name = meta["file_name"][i]
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        save_path = os.path.join(img_dir, base_name + ".png")

        Image.fromarray(img).save(save_path)


import os
import torch


def save_gt(dataname, meta, task_gts, task, pred_dir):
    """
    Save ground-truth images/maps for a given task.

    :param str dataname: Dataset name
    :param dict meta: Metadata from the dataset, containing image names and sizes
    :param dict task_gts: Ground-truth labels, e.g. batch["label"]
    :param str task: Task name, such as "edge", "semseg", "human_parts", etc.
    :param str pred_dir: Root directory to save results
    """
    import os
    import torch

    assert task in task_gts, f"Task '{task}' not found in task_gts."
    gt_root = os.path.join(pred_dir, "gt")
    gt_task = task_gts[task]
    gts = []

    for i in range(gt_task.size(0)):
        gt = gt_task[i]
        gt_task_root = os.path.join(gt_root,task)
        os.makedirs(gt_task_root, exist_ok=True)
        # -----------------------------
        # 1. 先整理成 save_img 期望的格式
        # -----------------------------
        if task in {"semseg", "human_parts"}:
            # 语义标签图必须是 [H, W]
            if gt.dim() == 3:
                if gt.shape[0] == 1:
                    gt = gt.squeeze(0)          # [1, H, W] -> [H, W]
                else:
                    gt = torch.argmax(gt, dim=0)  # 万一是 one-hot/logits

        elif task in {"edge", "sal", "depth"}:
            # 单通道连续图，保存时希望是 [H, W]
            if gt.dim() == 3 and gt.shape[0] == 1:
                gt = gt.squeeze(0)

        elif task == "normals":
            # 法线图要 [H, W, 3]
            if gt.dim() == 3 and gt.shape[0] == 3:
                gt = gt.permute(1, 2, 0)

        else:
            # 保守 fallback
            if gt.dim() == 3 and gt.shape[0] == 1:
                gt = gt.squeeze(0)

        # -----------------------------
        # 2. 裁掉 padding，和 predict() 保持一致
        # -----------------------------
        ori_dim = (int(meta["size"][i][0]), int(meta["size"][i][1]))  # (H, W)
        curr_dim = tuple(gt.shape[:2])

        if ori_dim != curr_dim:
            delta_h = max(curr_dim[0] - ori_dim[0], 0)
            delta_w = max(curr_dim[1] - ori_dim[1], 0)

            loc_h = [delta_h // 2, (delta_h // 2) + ori_dim[0]]
            loc_w = [delta_w // 2, (delta_w // 2) + ori_dim[1]]

            gt = gt[loc_h[0]:loc_h[1], loc_w[0]:loc_w[1]]

        gt = gt.detach().cpu().numpy()
        gts.append(gt)



    save_img(dataname, meta["file_name"], gts, task, gt_root)
