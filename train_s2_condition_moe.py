import argparse
import datetime
import os
import shutil

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE, NUM_TRAIN_IMAGES
from evaluation.evaluate_utils import PerformanceMeter
from losses import get_criterion
from models.condition_moe_mt_distiller import Condition_MoE_MTMT_Distiller
# from models.mt_distiller import MTMT_Distiller
from train_utils import cal_params, get_optimizer_scheduler,get_optimizer_scheduler_by_param_groups, update_weights, update_s2_weights,freeze_experts_by_indices, create_s2_param_groups_with_logging, create_s2_param_groups_full_decay, reset_router_bias
from utils import RunningMeter, create_results_dir, get_loss_metric, get_output, global_print, set_seed, to_cuda, bool_flag, log_config, load_checkpoint_state
from datasets.utils.configs import get_output_num
# ================= 1. 定义基线数据 (来自论文表格) =================
# 用于计算 Delta_m，注意：这里不包含 Edge/Boundary，因为训练时只有 Loss
BASELINES = {
    # 对应 PASCAL-Context (Table 5, ViT-L based)
    'pascal': {
        'semseg_mIoU': 80.25,       # ↑
        'human_parts_mIoU': 70.54,  # ↑
        'sal_maxF': 84.54,          # ↑
        'normals_mErr': 13.57       # ↓
        # edge_odsF: 74.22
    },
    # 对应 NYUD-v2 (Table 14, ViT-B based)
    'nyud': {
        'semseg_mIoU': 51.15,       # ↑
        'depth_RMSE': 0.5792,       # ↓
        'normals_mErr': 19.77       # ↓
    },
    # 'nyud': {
    #     'semseg_mIoU': 51.15,  # ↑
    #     'depth_RMSE': 0.5792,  # ↓
    #     'normals_mErr': 19.77  # ↓
    # }
}

PRISM_REFERENCE = {
    # 对应 PASCAL-Context (Table 5, ViT-L based)
    'pascal': {
        'semseg_mIoU': 81.88,       # ↑
        'human_parts_mIoU': 74.30,  # ↑
        'sal_maxF': 85.31,          # ↑
        'normals_mErr': 13.47       # ↓
        # edge_odsF: 74.69
    },
    # 对应 NYUD-v2 (Table 14, ViT-B based)
    'nyud': {
        'semseg_mIoU': 59.93,       # ↑
        'depth_RMSE': 0.4942,       # ↓
        'normals_mErr': 17.60       # ↓
        # 'edge_odsF': 78.60
    }
}
# 辅助函数：计算部分 Delta_m (排除 Edge)
def calc_partial_delta_m(dataset_name, current_metrics):
    # 简单的模糊匹配来确定使用哪套基线
    dataset_key = 'pascal' if 'pascal' in dataset_name.lower() else 'nyud'
    baselines = BASELINES.get(dataset_key)

    if not baselines:
        return -999  # 未知数据集

    gains = []
    # 遍历基线中的指标，如果在当前 logs 里能找到对应的，就计算 gain
    for metric, base_val in baselines.items():
        # 构造 log 中的 key (你的 log 带有 "eval/" 前缀)
        log_key = f"eval/{metric}"

        if log_key in current_metrics:
            curr_val = float(current_metrics[log_key])

            # 判断指标方向
            if 'mErr' in metric or 'RMSE' in metric:
                # 越低越好: (Base - Model) / Base
                gain = (base_val - curr_val) / base_val
            else:
                # 越高越好: (Model - Base) / Base
                gain = (curr_val - base_val) / base_val

            gains.append(gain)

    if not gains:
        return 0.0

    return (sum(gains) / len(gains)) * 100.0  # 返回百分比


# ================= 1. 增加：计算胜出数量的函数 =================
def calc_wins(dataset_name, current_metrics):
    """
    计算当前模型有多少个指标优于 Baseline (不含 Edge)
    """
    dataset_key = 'pascal' if 'pascal' in dataset_name.lower() else 'nyud'
    baselines = PRISM_REFERENCE.get(dataset_key)

    if not baselines:
        return 0, 0  # wins, total_checked

    wins = 0
    checked_count = 0

    for metric, base_val in baselines.items():
        log_key = f"eval/{metric}"

        if log_key in current_metrics:
            curr_val = float(current_metrics[log_key])
            checked_count += 1

            # 判断胜负
            if 'mErr' in metric or 'RMSE' in metric:
                # 越低越好: 当前值 < 基线值 = 赢
                if curr_val < base_val:
                    wins += 1
            else:
                # 越高越好: 当前值 > 基线值 = 赢
                if curr_val > base_val:
                    wins += 1

    return wins, checked_count


# ================= 2. 初始化最佳记录 =================
# ================= 2. 更新：初始化记录字典 =================
best_records = {
    'partial_delta_m': float('-inf'),
    'max_wins': -1,                   # 【新增】记录最多的胜出数量
    'max_wins_delta_m': float('-inf'),# 【新增】用于当胜出数量一样时，由 delta_m 决定谁更好
    'semseg_mIoU': float('-inf'),
    'human_parts_mIoU': float('-inf'),
    'depth_RMSE': float('inf'),
    'edge_loss': float('inf')
}

def train_one_iter_distiller(
    task_out,
    tasks,
    alpha,
    batch,
    model,
    optimizer,
    train_loss,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    optimizer.zero_grad()
    batch = to_cuda(batch)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
        if task_out:
            kd_loss_dict, aug_loss, task_loss_dict = model(batch)
            if kd_loss_dict is not None:
                kd_loss_dict["total"] = kd_loss_dict["total"] + aug_loss
        else:
            kd_loss_dict, aug_loss = model(batch)
            kd_loss_dict["total"] = kd_loss_dict["total"] + aug_loss

    # Log loss values
    batch_size = batch["image"].size(0)

    for task in tasks:
        loss_value = task_loss_dict[task].detach().item()
        train_loss[task].update(loss_value, batch_size)

    if kd_loss_dict is not None:
        for key in kd_loss_dict.keys():
            if key != "total":
                loss_value = kd_loss_dict[key].detach().item()
                train_kd_loss[key].update(loss_value, batch_size)
        if "decorrelation_loss" in kd_loss_dict:
            train_kd_loss["decorrelation_loss"].update(kd_loss_dict["decorrelation_loss"].detach().item(), batch_size)

        train_kd_loss["aug_loss"].update(aug_loss.detach().item(), batch_size)

        if task_out:
            scaler.scale(task_loss_dict["total"] + alpha * kd_loss_dict["total"]).backward()
        else:
            scaler.scale(kd_loss_dict["total"]).backward()
    else:
        if task_out:
            scaler.scale(task_loss_dict["total"]).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    for name, param in model.named_parameters():
        if torch.is_complex(param) or (param.grad is not None and torch.is_complex(param.grad)):
            print(f"发现复数参数/梯度: {name}")
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    epoch,
    iter_count,
    max_iter,
    task_out,
    tasks,
    alpha,
    train_dl,
    model,
    optimizer,
    scheduler,
    train_loss,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    train_dl.sampler.set_epoch(epoch)

    with tqdm(total=len(train_dl), disable=(int(os.environ["RANK"]) != 0)) as t:
        for batch in train_dl:

            t.set_description("Epoch: %d Iter: %d" % (epoch, iter_count))
            t.update(1)

            train_one_iter_distiller(
                task_out,
                tasks,
                alpha,
                batch,
                model,
                optimizer,
                train_loss,
                train_kd_loss,
                scaler,
                grad_clip,
                fp16,
            )

            if scheduler.__class__.__name__ == "PolynomialLR":
                scheduler.step()

            iter_count += 1

            if iter_count >= max_iter:
                end_signal = True
                break
            else:
                end_signal = False




    if scheduler.__class__.__name__ == "CosineLRScheduler":
        scheduler.step(epoch)

    return end_signal, iter_count


def eval_metric_rank0_only(task_out, tasks, dataname, val_dl, model, val_kd_loss):
    """
    Evaluate the model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            batch = to_cuda(batch)
            if task_out:
                kd_loss_dict, aux_loss, outputs = model.module.forward_val(batch)
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch["label"])
            else:
                kd_loss_dict, aux_loss = model.module.forward_val(batch)

            # Log loss values
            if isinstance(batch, dict):
                batch_size = batch["image"].size(0)
            else:
                batch_size = batch[0].size(0)

            for key in kd_loss_dict.keys():
                if key != "total":
                    loss_value = kd_loss_dict[key].detach().item()
                    val_kd_loss[key].update(loss_value, batch_size)
            val_kd_loss["aug_loss"].update(aux_loss.detach().item(), batch_size)


    results_dict = {}
    if task_out:
        eval_results = performance_meter.get_score()
        for task in tasks:
            for key in eval_results[task]:
                results_dict["eval/" + task + "_" + key] = eval_results[task][key]

    return results_dict


# def eval_metric(task_out, tasks, dataname, val_dl, model, val_kd_loss):
#     """
#     Evaluate the model in a distributed manner using a robust method (all_gather_object).
#     """
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     global_rank = dist.get_rank() if dist.is_initialized() else 0
#
#     # 重置所有 loss meter
#     for meter in val_kd_loss.values():
#         meter.reset()
#
#     model.eval()
#
#     # Step 1: 每个进程在自己的数据子集上进行评估，并收集结果到本地列表
#     local_predictions = {t: [] for t in tasks} if task_out else None
#     local_labels = {t: [] for t in tasks} if task_out else None
#
#     with torch.no_grad():
#         # 在主进程上显示 tqdm 进度条
#         pbar = tqdm(val_dl, desc="Evaluating", disable=(global_rank != 0))
#
#         for batch in pbar:
#             batch = to_cuda(batch)
#
#             if task_out:
#                 kd_loss_dict, aux_loss, outputs = model.module.forward_val(batch)
#
#                 # 收集预测和标签到 CPU，避免 GPU 内存累积
#                 for task in tasks:
#                     local_predictions[task].append(get_output(outputs[task], task).cpu())
#                     local_labels[task].append(batch["label"][task].cpu())
#             else:
#                 kd_loss_dict, aux_loss = model.module.forward_val(batch)
#
#             # 更新本地的 loss meter
#             batch_size = len(batch["image"])
#
#             for key, loss_val in kd_loss_dict.items():
#                 if key != "total":
#                     val_kd_loss[key].update(loss_val.item(), batch_size)
#             val_kd_loss["aug_loss"].update(aux_loss.item(), batch_size)
#
#
#     # 等待所有进程完成本地的评估循环
#     if dist.is_initialized():
#         dist.barrier()
#
#     # Step 2: 聚合所有进程的 loss 值
#     # -----------------------------------------------------------------
#     results_dict = {}
#     loss_keys = list(val_kd_loss.keys())
#     # 将所有本地 loss meter 的总和(sum)和计数(count)放入一个 tensor
#     local_loss_stats = torch.tensor([val_kd_loss[key].sum for key in loss_keys] +
#                                     [val_kd_loss[key].count for key in loss_keys]).cuda()
#
#     if dist.is_initialized():
#         dist.all_reduce(local_loss_stats, op=dist.ReduceOp.SUM)
#
#     if global_rank == 0:
#         global_sums = local_loss_stats[:len(loss_keys)]
#         global_counts = local_loss_stats[len(loss_keys):]
#         for i, key in enumerate(loss_keys):
#             # 避免除以零
#             avg_loss = global_sums[i] / global_counts[i] if global_counts[i] > 0 else 0
#             # 使用我之前建议的 val_KD_stats 格式
#             results_dict["eval_KD/" + key] = avg_loss.item()
#
#     # Step 3: 聚合任务指标的预测和标签到主进程并计算
#     # -----------------------------------------------------------------
#     if task_out:
#         # 为 all_gather_object 准备输出列表
#         gathered_preds_by_task = {t: [None] * world_size for t in tasks}
#         gathered_labels_by_task = {t: [None] * world_size for t in tasks}
#
#         if dist.is_initialized():
#             for task in tasks:
#                 # 收集每个任务的预测和标签
#                 dist.all_gather_object(gathered_preds_by_task[task], local_predictions[task])
#                 dist.all_gather_object(gathered_labels_by_task[task], local_labels[task])
#
#         if global_rank == 0:
#             performance_meter = PerformanceMeter(dataname, tasks)
#
#             # 在主进程上，将收集到的所有结果合并
#             # gathered_preds_by_task[task] 是一个列表的列表，需要先展平
#             all_preds = {t: torch.cat([item for sublist in gathered_preds_by_task[t] for item in sublist]) for t in
#                          tasks}
#             all_labels = {t: torch.cat([item for sublist in gathered_labels_by_task[t] for item in sublist]) for t in
#                           tasks}
#
#             # 使用完整的预测和标签更新 meter
#             # 假设 update 方法接收的是 {task: tensor} 格式的字典
#             performance_meter.update(all_preds, all_labels)
#
#             eval_results = performance_meter.get_score()
#             for task in tasks:
#                 for key, value in eval_results[task].items():
#                     results_dict["eval/" + task + "_" + key] = value
#
#     return results_dict

# def eval_metric(task_out, tasks, dataname, val_dl, model, val_kd_loss):
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     global_rank = dist.get_rank() if dist.is_initialized() else 0
#
#     for meter in val_kd_loss.values():
#         meter.reset()
#
#     model.eval()
#
#     local_predictions = {t: [] for t in tasks} if task_out else None
#     local_labels = {t: [] for t in tasks} if task_out else None
#
#     with torch.no_grad():
#         pbar = tqdm(val_dl, desc="Evaluating", disable=(global_rank != 0))
#         for batch in pbar:
#             batch = to_cuda(batch)
#             if task_out:
#                 kd_loss_dict, aux_loss, outputs = model.module.forward_val(batch)
#                 for task in tasks:
#                     # 将结果收集到CPU
#                     local_predictions[task].append(get_output(outputs[task], task).cpu())
#                     local_labels[task].append(batch["label"][task].cpu())
#             else:
#                 kd_loss_dict, aux_loss = model.module.forward_val(batch)
#
#             image_tensor = batch['image']
#             batch_size = image_tensor.size(0)
#             for key, loss_val in kd_loss_dict.items():
#                 if key != "total":
#                     val_kd_loss[key].update(loss_val.item(), batch_size)
#             val_kd_loss["aug_loss"].update(aux_loss.item(), batch_size)
#
#     if dist.is_initialized():
#         dist.barrier()
#
#     # --- Loss 聚合部分保持不变 ---
#     results_dict = {}
#     loss_keys = list(val_kd_loss.keys())
#     local_loss_stats = torch.tensor([val_kd_loss[key].sum for key in loss_keys] +
#                                     [val_kd_loss[key].count for key in loss_keys]).cuda()
#     if dist.is_initialized():
#         dist.all_reduce(local_loss_stats, op=dist.ReduceOp.SUM)
#
#     if global_rank == 0:
#         global_sums = local_loss_stats[:len(loss_keys)]
#         global_counts = local_loss_stats[len(loss_keys):]
#         for i, key in enumerate(loss_keys):
#             avg_loss = global_sums[i] / global_counts[i] if global_counts[i] > 0 else 0
#             results_dict["eval_KD/" + key] = avg_loss.item()
#     # -----------------------------
#
#     if task_out:
#         # --- 优化内存使用的聚合和计算 ---
#         gathered_preds_by_task = {t: None for t in tasks}
#         gathered_labels_by_task = {t: None for t in tasks}
#
#         for task in tasks:
#             if dist.is_initialized():
#                 # 创建接收列表
#                 gathered_preds = [None] * world_size
#                 gathered_labels = [None] * world_size
#                 # 聚合
#                 dist.all_gather_object(gathered_preds, local_predictions[task])
#                 dist.all_gather_object(gathered_labels, local_labels[task])
#                 # 聚合后，立即释放本地数据以节省内存
#                 del local_predictions[task]
#                 del local_labels[task]
#                 gathered_preds_by_task[task] = gathered_preds
#                 gathered_labels_by_task[task] = gathered_labels
#             else:  # 单GPU情况
#                 gathered_preds_by_task[task] = [local_predictions[task]]
#                 gathered_labels_by_task[task] = [local_labels[task]]
#
#         # 显式垃圾回收
#         del local_predictions
#         del local_labels
#         gc.collect()
#
#         if global_rank == 0:
#             performance_meter = PerformanceMeter(dataname, tasks)
#
#             all_preds_for_meter = {}
#             all_labels_for_meter = {}
#
#             for task in tasks:
#                 # 展平列表并拼接，然后立即释放中间列表
#                 pred_list = [item for sublist in gathered_preds_by_task[task] for item in sublist]
#                 all_preds_for_meter[task] = torch.cat(pred_list)
#                 del pred_list  # 释放内存
#
#                 label_list = [item for sublist in gathered_labels_by_task[task] for item in sublist]
#                 all_labels_for_meter[task] = torch.cat(label_list)
#                 del label_list  # 释放内存
#
#             # 显式垃圾回收
#             del gathered_preds_by_task
#             del gathered_labels_by_task
#             gc.collect()
#
#             # 使用拼接好的完整数据进行评估
#             performance_meter.update(all_preds_for_meter, all_labels_for_meter)
#
#             eval_results = performance_meter.get_score()
#             for task in tasks:
#                 for key, value in eval_results[task].items():
#                     results_dict["eval/" + task + "_" + key] = value
#         # ------------------------------------
#
#     return results_dict


# def eval_metric(task_out, tasks, dataname, val_dl, model, val_kd_loss):
#     """
#     Performs distributed evaluation.
#     - Updates task metrics via PerformanceMeter (which handles its own sync).
#     - Updates the externally provided val_kd_loss dictionary.
#     - Returns only the task metric results.
#     """
#     global_rank = dist.get_rank() if dist.is_initialized() else 0
#
#     # 1. 创建任务指标的 meter
#     performance_meter = PerformanceMeter(dataname, tasks if task_out else [])
#
#     # 2. 重置外部传入的 KD loss meters
#     for meter in val_kd_loss.values():
#         meter.reset()
#
#     # 3. 遍历数据并更新所有 meters
#
#     with torch.no_grad():
#         pbar = tqdm(val_dl, desc="Evaluating", disable=(global_rank != 0))
#         for batch in pbar:
#
#             batch = to_cuda(batch)
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 kd_loss_dict, aux_loss, outputs = model.module.forward_val(batch)
#
#             # 更新任务指标 meter
#             if task_out:
#                 predictions_for_meter = {t: get_output(outputs[t], t) for t in tasks}
#                 labels_for_meter = batch["label"]
#                 performance_meter.update(predictions_for_meter, labels_for_meter)
#
#             # 更新外部传入的 KD loss meters
#             batch_size = batch["image"].size(0)
#             if kd_loss_dict != None:
#                 for key, loss_value in kd_loss_dict.items():
#                     if key != "total" and key in val_kd_loss:
#                         val_kd_loss[key].update(loss_value.item(), batch_size)
#             if "aug_loss" in val_kd_loss:
#                 val_kd_loss["aug_loss"].update(aux_loss.item(), batch_size)
#
#
#     # 4. 获取任务指标结果 (内部已经实现了分布式同步)
#     # 这将是一个字典，且只在 rank 0 上有内容
#     task_results = performance_meter.get_score()
#     results_dict = {}
#     for task in tasks:
#         for key in task_results[task]:
#             results_dict["eval/" + task + "_" + key] = task_results[task][key]
#     return results_dict

def eval_metric(task_out, tasks, dataname, val_dl, model, val_kd_loss):
    """
    Performs distributed evaluation.
    - Updates task metrics via PerformanceMeter (which handles its own sync).
    - Updates the externally provided val_kd_loss dictionary.
    - Returns only the task metric results.
    """
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    performance_meter = PerformanceMeter(dataname, tasks if task_out else [])

    for meter in val_kd_loss.values():
        meter.reset()

    model.eval()

    with torch.no_grad():
        pbar = tqdm(val_dl, desc="Evaluating", disable=(global_rank != 0))
        for batch in pbar:
            batch = to_cuda(batch)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                kd_loss_dict, aux_loss, outputs = model.module.forward_val(batch)

            # 更新任务指标 meter
            if task_out:
                predictions_for_meter = {t: get_output(outputs[t], t) for t in tasks}
                labels_for_meter = batch["label"]
                performance_meter.update(predictions_for_meter, labels_for_meter)

            # 更新外部传入的 KD loss meters
            batch_size = batch["image"].size(0)
            if kd_loss_dict is not None:
                for key, loss_value in kd_loss_dict.items():
                    if key != "total" and key in val_kd_loss:
                        val_kd_loss[key].update(loss_value.item(), batch_size)

            if "aug_loss" in val_kd_loss:
                val_kd_loss["aug_loss"].update(aux_loss.item(), batch_size)

    task_results = performance_meter.get_score()
    results_dict = {}
    for task in tasks:
        for key in task_results[task]:
            results_dict["eval/" + task + "_" + key] = task_results[task][key]
    return results_dict


global_rank = int(os.environ["RANK"])

import datasets.custom_dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16")
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--task_out", action="store_true", help="Whether to output task or distill only")
    parser.add_argument("--alpha", type=float, default=1.0, help="Balance between task loss and distillation loss")

    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)

    # Join args and configs
    configs = {**configs, **vars(args)}

    # Set seed and ddp
    set_seed(args.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(0, 3600 * 2))
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    datasets.custom_dataset.N_TEACHERS = len(configs["teachers"])
    # Setup logger and output folders
    if global_rank == 0:
        log_config(configs)
        # print(configs)
        os.makedirs(configs["results_dir"], exist_ok=True)
        configs["exp_dir"] = create_results_dir(configs["results_dir"], args.exp)
        shutil.copy(args.config_path, os.path.join(configs["exp_dir"], "config.yml"))

    dist.barrier()

    # Setup dataset and dataloader
    dataname = configs["dataset"]
    task_dict = configs["task_dict"]
    task_list = []
    if args.task_out:
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]

    train_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=True)
    val_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=False)

    train_ds = get_dataset(dataname, train=True, tasks=task_list, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, drop_last=True)
    train_dl = get_dataloader(train=True, configs=configs, dataset=train_ds, sampler=train_sampler)

    # val_ds = get_dataset(dataname, train=False, tasks=task_list, transform=val_transforms)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False, drop_last=False)
    # # Pass the sampler to the dataloader. Also ensure the dataloader itself has shuffle=False.
    # val_dl = get_dataloader(train=False, configs=configs, dataset=val_ds, sampler=val_sampler)
    # # val_dl = get_dataloader(train=False, configs=configs, dataset=val_ds)

    val_ds = get_dataset(dataname, train=False, tasks=task_list, transform=val_transforms)
    # --- 必须使用 DistributedSampler ---
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False, drop_last=False)
    val_dl = get_dataloader(train=False, configs=configs, dataset=val_ds, sampler=val_sampler)

    # Setup loss function
    if args.task_out:
        criterion = get_criterion(dataname, task_list).cuda()
    else:
        criterion = None

    # Setup model
    model = Condition_MoE_MTMT_Distiller(
        img_size=INPUT_SIZE[dataname],
        tea_configs=configs["teachers"],
        stu_config=configs["student"],
        loss_type=configs["loss_type"],
        task_out=args.task_out,
        stu_criterion=criterion,
        dataname=dataname,
        tasks=task_list,
    ).cuda()

    if global_rank == 0:
        cal_params(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # base_lr_for_pretrained = float(configs["base_lr"]) * (float(configs["tr_batch"]) * dist.get_world_size() / 128)
    #
    # new_module_lr_multiplier = 5.0
    # # base_lr_for_pretrained = new_modules_lr / new_module_lr_multiplier  # 假设预训练模块的学习率是新模块的十分之一
    # # 2. 创建参数组
    # # 我们将新模块的学习率设为 new_modules_lr，然后计算出乘数
    #
    #
    # param_groups = create_s2_param_groups_with_logging(
    #     model=model,
    #     base_lr=base_lr_for_pretrained,
    #     weight_decay=float(configs["weight_decay"]),
    #     new_module_lr_multiplier=new_module_lr_multiplier,
    #     head_lr_multiplier=new_module_lr_multiplier  # 假设头和新模块使用相同的大学习率
    # )

    # param_groups = create_s2_param_groups_full_decay(
    #     model=model,
    #     base_lr=base_lr_for_pretrained,
    #     weight_decay=float(configs["weight_decay"]),
    #     new_module_lr_multiplier=new_module_lr_multiplier,
    #     head_lr_multiplier=new_module_lr_multiplier  # 假设头和新模块使用相同的大学习率
    # )
    # Setup optimizer and scheduler
    # optimizer, scheduler = get_optimizer_scheduler_by_param_groups(configs, param_groups, base_lr_for_pretrained, float(configs["weight_decay"]))

    optimizer, scheduler =  get_optimizer_scheduler(configs, model)
    # Setup scaler for amp
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # Setup loss meters
    train_loss = {}
    train_KD_loss = {}
    val_KD_loss = {}
    for task in task_list:
        train_loss[task] = RunningMeter()
    for i in range(len(configs["teacher_output_indices"])):
        for tea_no, _ in enumerate(configs["teachers"].items()):
            train_KD_loss[str(tea_no) + "_level" + str(i + 1)] = RunningMeter()
            val_KD_loss[str(tea_no) + "_level" + str(i + 1)] = RunningMeter()
    if "sam_grad" in configs["loss_type"]:
        for i in range(2):
            train_KD_loss[f"2_grad_loss_level{i + 1}"] = RunningMeter()
    train_KD_loss["aug_loss"] = RunningMeter()
    if "decorrelation" in configs["loss_type"]:
        train_KD_loss["decorrelation_loss"] = RunningMeter()
    val_KD_loss["aug_loss"] = RunningMeter()
    # Determine max epochs and iterations
    max_epochs = configs["max_epochs"]
    max_iter = configs["max_iters"]

    if max_epochs > 0:
        max_iter = 10000000
    else:
        assert max_iter > 0
        max_epochs = 1000000

    start_epoch = 0
    iter_count = 0

    if args.checkpoint is not None:
        global_print("Loading checkpoint from %s" % args.checkpoint)
        checkpoint, state_dict, _, _ = load_checkpoint_state(args.checkpoint, weights_only=True, print_fn=global_print)

        # Update student weights
        if not args.resume:
            update_s2_weights(model.module.student, state_dict)
        # model.module.student.backbone.eval()
        # freeze_experts_by_indices(model.module.student,
        #                           configs["student"].get("frozen_expert_indices", list(range(configs["student"]["backbone"]["num_moe_experts"]))),
        #                           freeze_shared_expert=False) # [0, 1, 2, 3, 4, 5, 6, 7]))
        # reset_router_bias(model)
        # model.module.student.backbone.vfm_condition_embeddings.requires_grad_(False)
        with torch.no_grad():
            # 假设 0:DINO, 1:CLIP, 2:SAM
            vfm_embs = model.module.student.backbone.vfm_condition_embeddings.weight
            task_embs = model.module.student.backbone.task_condition_embeddings.weight
              # print("!!!!!!!!!!!!!!!!!!!!! init task embeddings from vfm embeddings !!!!!!!!!!!!!!!!!!!!!")
            # 强制赋值（建立先验）
            # Semseg (假设是 task 0) -> DINO
            # task_embs[0].copy_(vfm_embs[0])
            # Human Parts (假设是 task 1) -> DINO
            # task_embs[1].copy_(vfm_embs[0])
            # global_print("#### normals -> DINO  ####")#
            # task_embs[2].copy_(vfm_embs[0])

            global_print("####  Edge -> SAM  ####")#
            task_embs[3].copy_(vfm_embs[0])
            # # Saliency (假设是 task 4) -> CLIP
            # global_print("##### Saliency -> CLIP ####")  #
            # task_embs[4].copy_(vfm_embs[1])


            # # 强制赋值（建立先验）
            # Semseg (假设是 task 0) -> DINO
            # task_embs[0].copy_(vfm_embs[0])
            # normals (假设是 task 1) -> DINO
            # task_embs[1].copy_(0.5 * vfm_embs[0])
            # Edge (假设是 task 2) -> SAM
            # task_embs[2].copy_(vfm_embs[2])
            # depth (假设是 task 3) -> SAM
            # task_embs[3].copy_(vfm_embs[1])


        if args.resume:
            model_state = state_dict

            clean_model_state = {}
            for k, v in model_state.items():
                if torch.is_complex(v):
                    print(f"🔧 修复损坏的参数: {k} (Complex -> Real)")
                    clean_model_state[k] = v.real.float()  # 强行转回实数
                else:
                    clean_model_state[k] = v
            model.module.student.load_state_dict(clean_model_state)

            # 2. ⚠️ 关键：不要加载 Optimizer State
            # optimizer.load_state_dict(checkpoint['optimizer'])  <-- 这一行一定要注释掉！
            print("已丢弃旧的优化器状态 (Optimizer State)，以消除潜在的复数污染。")

            # 3. 重新初始化优化器
            # 就像从头开始训练一样初始化它，这样它的动量缓存是空的，绝对干净。
            # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # 这里手动指定一个较小的 LR
            print("优化器已重置，学习率设置为: 1e-5")
            # if "optimizer" in checkpoint.keys():
            #     optimizer.load_state_dict(checkpoint["optimizer"])
            # if "scheduler" in checkpoint.keys():
            #     scheduler.load_state_dict(checkpoint["scheduler"])
            if isinstance(checkpoint, dict) and "epoch" in checkpoint.keys():
                start_epoch = checkpoint["epoch"] + 1
            if isinstance(checkpoint, dict) and "iter_count" in checkpoint.keys():
                iter_count = checkpoint["iter_count"]
            # if "model" in checkpoint.keys():
            #     model_state_dict = checkpoint["model"]
            #     model.module.student.load_state_dict(model_state_dict, strict=True)

    global_print(
        "Start: Epoch %d, Iter %d, Goal: Epoch %d or Iter %d" % (start_epoch, iter_count, max_epochs, max_iter)
    )

    for epoch in range(start_epoch, max_epochs):
        if dist.is_initialized():
            dist.barrier()
        end_signal, iter_count = train_one_epoch(
            epoch,
            iter_count,
            max_iter,
            args.task_out,
            task_list,
            args.alpha,
            train_dl,
            model,
            optimizer,
            scheduler,
            train_loss,
            train_KD_loss,
            scaler,
            configs["grad_clip"],
            args.fp16,
        )

        # Validation
        # if global_rank == 0:
        if (epoch + 1) % configs["eval_freq"] == 0 or epoch == max_epochs - 1 or end_signal:
            # Save checkpoint
            model.eval()


            if global_rank == 0:
                save_ckpt_temp = {
                    "model": model.module.student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iter_count": iter_count,
                }
                torch.save(
                    save_ckpt_temp,
                    os.path.join(configs["exp_dir"], "checkpoint.pth"),
                )
                global_print("Checkpoint saved.")

                global_print("Validation at epoch %d." % epoch)

            with torch.no_grad():
                val_logs = eval_metric(args.task_out, task_list, dataname, val_dl, model, val_KD_loss)
                val_KD_stats = get_loss_metric(val_KD_loss, val_KD_loss.keys(), "eval_KD")
                val_logs.update(val_KD_stats)

            if global_rank == 0:
                global_print(val_logs)

                # --- 提取当前指标值 (处理 float64) ---
                curr_semseg = float(val_logs.get('eval/semseg_mIoU', -1))
                curr_edge_loss = float(val_logs.get('eval/edge_loss', 999))

                # --- 计算 Partial Delta_m (核心逻辑) ---
                # 自动根据 dataname 选择基线，计算不含 Edge 的平均相对提升

                # ================= 保存策略 =================
                # 获取当前数值
                curr_delta_m = calc_partial_delta_m(dataname, val_logs)
                curr_wins, total_tasks = calc_wins(dataname, val_logs)  # 计算胜出数

                # -----------------------------------------------------------
                # 【策略 A】：保存 Delta_m 最高的 (平均性能最强)
                # -----------------------------------------------------------
                if curr_delta_m > best_records['partial_delta_m']:
                    best_records['partial_delta_m'] = curr_delta_m
                    torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_delta_m.pth"))
                    global_print(f"★ [Epoch {epoch}] New Best Avg Delta_m: {curr_delta_m:.2f}%")

                # -----------------------------------------------------------
                # 【策略 B (新增)】：保存“击败Baseline项目最多”的模型
                # -----------------------------------------------------------
                # 逻辑：
                # 1. 如果当前胜出数 > 历史最高胜出数 -> 绝对保存
                # 2. 如果当前胜出数 == 历史最高胜出数，但在平均分(delta_m)上更高 -> 替换保存 (更优的平局)
                if (curr_wins > best_records['max_wins']) or \
                        (curr_wins == best_records['max_wins'] and curr_delta_m > best_records['max_wins_delta_m']):
                    best_records['max_wins'] = curr_wins
                    best_records['max_wins_delta_m'] = curr_delta_m  # 更新对应的 delta_m

                    torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_most_wins.pth"))
                    global_print(
                        f"★ [Epoch {epoch}] New Most Wins Model! Wins: {curr_wins}/{total_tasks} (Avg Gain: {curr_delta_m:.2f}%)")



                # 2. 保存 SemSeg 最好的模型 (两个数据集通用)
                if curr_semseg > best_records['semseg_mIoU']:
                    best_records['semseg_mIoU'] = curr_semseg
                    torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_semseg.pth"))
                    global_print(f"★ [Epoch {epoch}]  New Best SemSeg: {curr_semseg:.2f}")

                # 3. 保存 Edge Loss 最低的模型 (用于后续离线算 ODS)
                if curr_edge_loss < best_records['edge_loss']:
                    best_records['edge_loss'] = curr_edge_loss
                    torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_edge_loss.pth"))
                    global_print(f"★ [Epoch {epoch}]  New Lowest Edge Loss: {curr_edge_loss:.4f}")

                # 4. 数据集特有指标保存
                if 'pascal' in dataname.lower():
                    # 保存 Human Parsing
                    curr_human = float(val_logs.get('eval/human_parts_mIoU', -1))
                    if curr_human > best_records['human_parts_mIoU']:
                        best_records['human_parts_mIoU'] = curr_human
                        torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_human_parsing.pth"))
                        global_print(f"★ [Epoch {epoch}]  New Best Human Parts mIoU: {curr_human:.2f}")

                elif 'nyud' in dataname.lower():
                    # 保存 Depth (RMSE 越低越好)
                    curr_depth = float(val_logs.get('eval/depth_RMSE', 999))
                    if curr_depth < best_records['depth_RMSE']:
                        best_records['depth_RMSE'] = curr_depth
                        torch.save(save_ckpt_temp, os.path.join(configs["exp_dir"], "best_depth.pth"))
                        global_print(f"★ [Epoch {epoch}]  New Best Depth RMSE: {curr_depth:.4f}")
            dist.barrier()
            model.train()
        if end_signal:
            break

    global_print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
