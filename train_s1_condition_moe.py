import argparse
import datetime
import os
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE
from models.condition_moe_mt_distiller import Condition_MoE_MTMT_Distiller
from train_utils import cal_params, update_weights, create_s1_param_groups_with_logging
from utils import RunningMeter, create_results_dir, get_loss_metric, global_print, set_seed, to_cuda, bool_flag, log_config, load_checkpoint_state
import gc

def get_diversity_weight(step, start_step=1625, warmup_steps=2437, decline_steps=3249, peak_weight=20.0, final_weight=1.0):
    if step < start_step:
        return 0.0
    elif step < warmup_steps:
        # 线性增长到峰值
        return peak_weight * (step / warmup_steps)
    elif step < decline_steps:
        # 线性下降到最终值
        return peak_weight - (peak_weight - final_weight) * ((step - warmup_steps) / (decline_steps - warmup_steps))
    else:

        # 保持在最终的低权重
        return final_weight


def train_one_iter_distiller(
    batch,
    model,
    optimizer,
    train_kd_loss,
    scaler,
    grad_clip,
    use_bf16,
    step=-1,
):
    optimizer.zero_grad()
    batch = to_cuda(batch)
    # Use the check from Step 1

    with torch.autocast(device_type="cuda", enabled = use_bf16, dtype = torch.bfloat16 ): #, dtype=torch.float16, enabled=fp16): #
    # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True): #):
    # with torch.cuda.amp.autocast(fp16 is not None):
        kd_loss_dict, aug_loss, gate_loss = model(batch, step=step)
        # print(step)

        gate_loss = get_diversity_weight(step) * gate_loss
        gate_loss = torch.clamp(gate_loss, min=-0.5)
        gate_loss = torch.tensor(0.0).to(aug_loss.device)
        kd_loss_dict["total"] = kd_loss_dict["total"] + aug_loss +gate_loss
    # Log loss values
    batch_size = batch["image"].size(0)

    for key in kd_loss_dict.keys():
        if key != "total":
            loss_value = kd_loss_dict[key].detach().item()
            train_kd_loss[key].update(loss_value, batch_size)

    train_kd_loss["aug_loss"].update(aug_loss.detach().item(), batch_size)
    train_kd_loss["gate_loss"].update(gate_loss.detach().item(), batch_size)
    # if "gate_loss" in kd_loss_dict:
    #     train_kd_loss["gate_loss"].update(gate_loss.detach().item(), batch_size)

    if "decorrelation_loss" in kd_loss_dict:
        train_kd_loss["decorrelation_loss"].update(kd_loss_dict["decorrelation_loss"].detach().item(), batch_size)
    scaler.scale(kd_loss_dict["total"]).backward()
    # Standard backward pass. No more scaler!
    # total_loss = kd_loss_dict["total"]
    # total_loss.backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    # optimizer.step()
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    epoch,
    iter_count,
    train_dl,
    model,
    optimizer,
    scheduler,
    train_kd_loss,
    scaler,
    grad_clip,
    use_bf16,
):
    train_dl.sampler.set_epoch(epoch)

    with tqdm(total=len(train_dl), disable=(int(os.environ["RANK"]) != 0)) as t:
        for batch in train_dl:
            t.set_description("Epoch: %d " % (epoch))
            t.update(1)

            train_one_iter_distiller(
                batch,
                model,
                optimizer,
                train_kd_loss,
                scaler,
                grad_clip,
                use_bf16,
                step = iter_count
            )

            scheduler.step(iter_count)
            iter_count += 1

    return iter_count

import datasets.custom_dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=bool_flag, default=True, help="Whether to use fp16")

    parser.add_argument("--checkpoint", default=None, help="Load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # args.fp16 = torch.cuda.amp.GradScaler()
    args = parser.parse_args()
    use_bf16 = torch.cuda.is_bf16_supported()
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

    # datasets.custom_dataset.N_TEACHERS = len(configs["teachers"])
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
    task_list = []

    train_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=True)
    train_ds = get_dataset(dataname, train=True, tasks=task_list, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=True)
    # train_sampler = TeacherPartitionSampler(
    #     dataset,
    #     batch_per_teacher=BATCH_PER_TEACHER,
    #     drop_last=True,  # 通常训练时为True
    #     shuffle=True,  # 通常训练时为True
    #     ddp_rank=ddp_rank,
    #     ddp_world_size=ddp_world_size
    # )
    train_dl = get_dataloader(train=True, configs=configs, dataset=train_ds, sampler=train_sampler)

    # Setup model
    model = Condition_MoE_MTMT_Distiller(
        img_size=INPUT_SIZE[dataname],
        tea_configs=configs["teachers"],
        stu_config=configs["student"],
        loss_type=configs["loss_type"],
        task_out=False,
    ).cuda()
    if global_rank == 0:
        cal_params(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    # print("\n--- Checking parameter devices ---")
    # for name, param in model.named_parameters():
    #     if "backbone" in name:  # 只看和backbone相关的参数
    #         print(f"Parameter: {name:<50} Device: {param.device}")


    model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    scaled_lr = float(configs["base_lr"]) * (float(configs["tr_batch"]) * dist.get_world_size() / 128)
    new_modules_lr = min(scaled_lr, 1e-3)
    base_lr_for_pretrained = new_modules_lr / 100.0
    # 2. 创建参数组
    # 我们将新模块的学习率设为 new_modules_lr，然后计算出乘数
    new_module_lr_multiplier = 100.0

    param_groups = create_s1_param_groups_with_logging(
        model=model,
        base_lr=base_lr_for_pretrained,
        weight_decay=float(configs["weight_decay"]),
        new_module_lr_multiplier=new_module_lr_multiplier,
        head_lr_multiplier=new_module_lr_multiplier  # 假设头和新模块使用相同的大学习率
    )
    optimizer = torch.optim.AdamW(
        params=param_groups,
        # lr 和 weight_decay 已经在 param_groups 中定义，这里可以省略
        # 但为了安全，PyTorch会使用这里的值作为默认值，所以写上也没问题
        lr=new_modules_lr,
        weight_decay=float(configs["weight_decay"])
    )

    # # Setup optimizer and scheduler
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=float(configs["base_lr"]) * (float(configs["tr_batch"]) * dist.get_world_size() / 128),
    #     weight_decay=float(configs["weight_decay"]),
    # )
    max_epochs = int(configs["max_epochs"])
    warmup_epochs = int(configs["warmup_epochs"])
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(max_epochs - warmup_epochs) * len(train_dl),
        lr_min=1e-6, #1e-5
        warmup_t=warmup_epochs * len(train_dl),
        warmup_lr_init=1e-7, #1.25e-7
        warmup_prefix=True,
    )
    # global_print(use_bf16)
    # Setup scaler for amp
    scaler = torch.amp.GradScaler(enabled=use_bf16)
    # scaler = torch.amp.GradScaler(enabled=True)
    # scaler = None
    # Setup loss meters
    train_KD_loss = {}
    for i in range(len(configs["teacher_output_indices"])):
        for tea_no, _ in enumerate(configs["teachers"].items()):
            train_KD_loss[str(tea_no) + "_level" + str(i + 1)] = RunningMeter()
    if "sam_grad" in configs["loss_type"]:
        for i in range(2):
            train_KD_loss[f"2_grad_loss_level{i + 1}"] = RunningMeter()
    train_KD_loss["aug_loss"] = RunningMeter()
    # if configs["student"]["backbone"]["moe_type"] == "gated":
    train_KD_loss["gate_loss"] = RunningMeter()
    if "decorrelation" in configs["loss_type"]:
        train_KD_loss["decorrelation_loss"] = RunningMeter()
    # Determine max epochs and iterations
    max_epochs = configs["max_epochs"]
    start_epoch = 0
    iter_count = 0


    if args.checkpoint is not None:
        # Resume training
        global_print("################### Loading checkpoint from %s ######################" % args.checkpoint)
        checkpoint, state_dict, _, _ = load_checkpoint_state(args.checkpoint, weights_only=True, print_fn=global_print)

        # Update student weights
        update_weights(model.module.student, state_dict)

        if args.resume and isinstance(checkpoint, dict):
            if "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint.keys():
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint.keys():
                start_epoch = checkpoint["epoch"] + 1
            if "iter_count" in checkpoint.keys():
                iter_count = checkpoint["iter_count"]

        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()

    global_print("Start: Epoch %d, Iter %d, Goal: Epoch %d" % (start_epoch, iter_count, max_epochs))

    for epoch in range(start_epoch, max_epochs):
        iter_count = train_one_epoch(
            epoch,
            iter_count,
            train_dl,
            model,
            optimizer,
            scheduler,
            train_KD_loss,
            scaler,
            configs["grad_clip"],
            use_bf16, #args.fp16,
        )

        # Save checkpoint
        if global_rank == 0:
            if (epoch % configs["save_freq"] == 0 or (epoch + 1) == max_epochs):
                # Save checkpoint
                save_ckpt_temp = {
                    "model": model.module.student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iter_count": iter_count,
                }
                torch.save(
                    save_ckpt_temp,
                    os.path.join(configs["exp_dir"], str(epoch) + "_checkpoint.pth"),
                )
                global_print("Checkpoint saved.")
            else:
                # Save checkpoint
                save_ckpt_temp = {
                    "model": model.module.student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iter_count": iter_count,
                }
                torch.save(
                    save_ckpt_temp,
                    os.path.join(configs["exp_dir"], "latest_checkpoint.pth"),
                )
                global_print(str(epoch) + "epoch checkpoint saved to latest_checkpoint.pth.")

        dist.barrier()

    global_print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
