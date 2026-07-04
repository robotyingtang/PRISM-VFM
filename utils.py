import os

import numpy as np
import torch
import torch.nn.functional as F


class RunningMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_dir(directory):
    """
    Create required directory if it does not exist
    """

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_results_dir(results_dir, exp_name):
    """
    Create required results directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :return: Path of experiment directory and checkpoint directory
    """

    exp_dir = os.path.join(results_dir, exp_name)
    create_dir(results_dir)
    create_dir(exp_dir)

    return exp_dir


def create_pred_dir(results_dir, exp_name, tasks):
    """
    Create required prediction directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :param list tasks: List of tasks
    :return: Path of checkpoint directory and prediction dictionary
    """

    exp_dir = os.path.join(results_dir, exp_name)
    pred_dir = os.path.join(exp_dir, "predictions")
    create_dir(pred_dir)

    for task in tasks:
        task_dir = os.path.join(pred_dir, task)
        create_dir(task_dir)
        if task == "edge":
            create_dir(os.path.join(task_dir, "img"))

    return exp_dir, pred_dir


def get_loss_metric(loss_meter, tasks, prefix):
    """
    Get loss statistics
    :param dict loss_meter: Loss meter
    :param str tasks: List of tasks
    :param str prefix: Prefix for the loss, train or val
    :return: Loss statistics
    """

    statistics = {prefix + "/" + "loss_sum": 0.0}

    for task in tasks:
        statistics[prefix + "/" + "loss_sum"] += loss_meter[task].avg
        statistics[prefix + "/" + task] = loss_meter[task].avg
        loss_meter[task].reset()

    return statistics


def to_cuda(batch):
    """
    Move batch to GPU
    :param dict batch: Input batch
    :return: Batch on GPU
    """

    if type(batch) is dict:
        out = {}
        for k, v in batch.items():
            if k == "meta":
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) is torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) is list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


def get_output(output, task):
    """
    Get output prediction in the required range and format
    :param Tensor output: Output tensor
    :param str task: Task
    :return: Tensor
    """

    if task == "normals":
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {"semseg", "human_parts"}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {"edge"}:
        output = output.permute(0, 2, 3, 1)
        output = torch.sigmoid(output).squeeze(-1) * 255

    elif task in {"sal"}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255

    elif task in {"depth"}:
        output.clamp_(min=0.0)
        output = output.permute(0, 2, 3, 1).squeeze(-1)

    else:
        raise NotImplementedError

    return output


def global_print(s):
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) == 0:
            print(s)
    else:
        print(s)


def log_config(config_obj):
    """记录配置信息的通用函数"""
    config_dict = {}

    # 处理不同类型的配置对象
    if isinstance(config_obj, dict):
        config_dict = config_obj
    elif hasattr(config_obj, '__dict__'):
        config_dict = vars(config_obj)
    elif hasattr(config_obj, '_asdict'):  # 处理 namedtuple
        config_dict = config_obj._asdict()
    else:
        # 尝试其他可能的转换方式
        try:
            config_dict = dict(config_obj)
        except (TypeError, ValueError):
            print("无法转换配置对象为字典")
            return

    for k, v in sorted(config_dict.items()):
        # 跳过私有属性
        if not k.startswith('_'):
            print("\t{}: {}".format(k, str(v)))

def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


##################################################
# Utility routines (functions and classes) used for training models.
# Some of these routines are re-used from
# - DINO (https://github.com/facebookresearch/dino)
# - MoCo (https://github.com/facebookresearch/moco)
# - PyTorch examples (https://github.com/pytorch/examples)
##################################################
###############################from UNIC

import argparse
import os
import random
import sys
import logging
import json
import pickle
from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

# from dinov2 import distributed


logger = logging.getLogger()


def save_pickle(obj, save_path):
    with open(save_path, "wb") as fid:
        pickle.dump(obj, fid)


def load_pickle(save_path):
    with open(save_path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def torch_load_cpu(path, weights_only=True):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_state(path, weights_only=True, print_fn=global_print):
    checkpoint = torch_load_cpu(path, weights_only=weights_only)

    if isinstance(checkpoint, dict):
        top_keys = sorted(checkpoint.keys())
        meta = checkpoint.get("meta")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        top_keys = []
        meta = None
        state_dict = checkpoint

    if print_fn is not None:
        print_fn(f"Checkpoint top-level keys: {top_keys}")
        if isinstance(meta, dict):
            print_fn(f"Checkpoint meta keys: {sorted(meta.keys())}")
            if "removed_top_level_keys" in meta:
                print_fn(f"Checkpoint removed_top_level_keys: {meta['removed_top_level_keys']}")

    return checkpoint, state_dict, meta, top_keys


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

        if "RANK" in os.environ:
            args.rank = int(os.environ["RANK"])
        elif "SLURM_PROCID" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
        else:
            print("Cannot find rank in environment variables")
            sys.exit(-1)

        n_gpus_per_node = torch.cuda.device_count()
        assert n_gpus_per_node > 0, "No GPU device detected"

        args.gpu = args.rank - n_gpus_per_node * (args.rank // n_gpus_per_node)

    # launched naively with "python main.py"
    elif torch.cuda.is_available():
        print("==> Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12345"

    else:
        print("==> Does not support training without GPU.")
        sys.exit(1)

    print(
        "=> WORLD_SIZE={}, RANK={}, GPU={}, MASTER_ADDR={}, MASTER_PORT={}, INIT_METHOD={}".format(
            args.world_size,
            args.rank,
            args.gpu,
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            args.dist_url,
        ),
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def fix_random_seeds(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_program_info(args):
    logger.info("Args:")
    for k, v in sorted(dict(vars(args)).items()):
        logger.info("\t{}: {}".format(k, str(v)))

    with open(os.path.join(args.output_dir, "args.json"), "w") as fp:
        json.dump(
            dict(vars(args)),
            fp,
            indent=4,
            sort_keys=True,
        )

    logger.info("Env vars:")
    for env_var in [
        "ONEDAL_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "MYDEBUG",
    ]:
        logger.info("\t{}={}".format(env_var, os.environ.get(env_var, "(unset)")))

    logger.info("Script caller: {}".format(sys.argv[0]))
    for parg in sys.argv[1:]:
        logger.info("\t{}".format(parg))


def save_model_defn(model, save_path):
    fp = open(os.path.join(save_path), "w")
    fp.write("{}".format(model))
    fp.write("\n")

    modules = {
        "model": model,
        "encoder": model.encoder,
        "lp": model.lp,
    }

    for mname, module in modules.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        fp.write(
            "Number of trainable parameters in {} : {:,}\n".format(mname, trainable)
        )
        fp.write("Number of frozen parameters in {} : {:,}\n".format(mname, frozen))

    fp.flush()
    fp.close()

def save_model_defn_decoder(model, save_path):
    fp = open(os.path.join(save_path), "w")
    fp.write("{}".format(model))
    fp.write("\n")

    modules = {
        "model": model,
        "decoder": model.decoder,
        "repara_module": model.repara_module,
        "dec2teacher_module": model.dec2teacher_module,
    }

    for mname, module in modules.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        fp.write(
            "Number of trainable parameters in {} : {:,}\n".format(mname, trainable)
        )
        fp.write("Number of frozen parameters in {} : {:,}\n".format(mname, frozen))

    fp.flush()
    fp.close()

def save_model_defn_encoder(model, save_path):
    fp = open(os.path.join(save_path), "w")
    fp.write("{}".format(model))
    fp.write("\n")

    modules = {
        "model": model,
        "encoder": model.encoder,
        "repara_module": model.repara_module,
    }

    for mname, module in modules.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        fp.write(
            "Number of trainable parameters in {} : {:,}\n".format(mname, trainable)
        )
        fp.write("Number of frozen parameters in {} : {:,}\n".format(mname, frozen))

    fp.flush()
    fp.close()


def get_encoder_params_groups(
        model,
        decoder,
        save_file_path=None,
        encoder_lr_scale=1.0,
        head_lr_scale=1.0,
        decoder_lr_scale=0.1,
        wd=0.05):
    """
    改进后的参数分组函数，支持：
    1. 区分encoder和mlp head参数
    2. 分别设置学习率比例
    3. 保留原有正则化分组逻辑
    """
    encoder_regularized = []
    encoder_not_regularized = []
    head_regularized = []
    head_not_regularized = []
    decoder_regularized = []  # 其他模块参数
    decoder_not_regularized = []

    fp = None
    if save_file_path is not None:
        fp = open(save_file_path, "w")


    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # 判断参数归属模块
        if "encoder" in name:  # encoder参数
            module_type = "encoder"
            if name.endswith(".bias") or len(param.shape) == 1:
                encoder_not_regularized.append(param)
                regstat = "Encoder_NotRegularized"
            else:
                encoder_regularized.append(param)
                regstat = "Encoder_Regularized"

        elif "repara_module" in name:  # mlp head参数
            module_type = "repara_module"
            if name.endswith(".bias") or len(param.shape) == 1:
                head_not_regularized.append(param)
                regstat = "Head_NotRegularized"
            else:
                head_regularized.append(param)
                regstat = "Head_Regularized"

        # else:  # 其他模块参数
        #     module_type = "other"
        #     if name.endswith(".bias") or len(param.shape) == 1:
        #         reg_group = other_not_regularized
        #         regstat = "Other_NotRegularized"
        #     else:
        #         reg_group = other_regularized
        #         regstat = "Other_Regularized"

        # reg_group.append(param)

        if fp is not None:
            fp.write("{} - {} - {}\n".format(name, list(param.shape), regstat))


    for name, param in decoder.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            regstat = "Decoder_NotRegularized"
            decoder_not_regularized.append(param)
        else:
            regstat = "Decoder_Regularized"
            decoder_regularized.append(param)

        if fp is not None:
            fp.write("{} - {} - {}\n".format(name, list(param.shape), regstat))


    if fp is not None:
        fp.close()

    return [
        {"params": encoder_regularized, "weight_decay": wd, "lr_scale": encoder_lr_scale},
        {"params": encoder_not_regularized, "weight_decay": 0.0, "lr_scale": encoder_lr_scale},
        {"params": head_regularized, "weight_decay": wd, "lr_scale": head_lr_scale},
        {"params": head_not_regularized, "weight_decay": 0.0, "lr_scale": head_lr_scale},
        {"params": decoder_regularized, "weight_decay": wd, "lr_scale": decoder_lr_scale},  # 其他模块默认学习率比例1.0
        {"params": decoder_not_regularized, "weight_decay": 0.0, "lr_scale": decoder_lr_scale},
    ]


def get_params_groups(model, save_file_path=None):
    """
    Returns two parameters group, one for regularized parameters with weight decay,
    and another for unregularized parameters.
    """
    regularized = []
    not_regularized = []

    fp = None
    if save_file_path is not None:
        fp = open(save_file_path, "w")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            regstat = "Not Regularized"
            not_regularized.append(param)
        else:
            regstat = "Regularized"
            regularized.append(param)

        if fp is not None:
            fp.write("{} - {} - {}\n".format(name, list(param.shape), regstat))

    if fp is not None:
        fp.flush()
        fp.close()

    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    Creates a cosine scheduler with linear warm-up.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(p=2)
            norms.append(param_norm)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return torch.stack(norms)


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        logger.info("Have not found checkpoint at {}".format(ckp_path))
        return
    logger.info("Found checkpoint at {}".format(ckp_path))
    checkpoint = torch.load(ckp_path, map_location="cpu")

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logger.info(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logger.info(
                        "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                    )
                except ValueError:
                    logger.info(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            logger.info(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                var = checkpoint[var_name]
                var = move_tensors_to_cuda(var)
                run_variables[var_name] = var


def move_tensors_to_cuda(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_tensors_to_cuda(value)
    elif isinstance(obj, (list, tuple)):
        obj = [move_tensors_to_cuda(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.cuda()
    return obj


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def standard_normalize(data, mean_ema=None, std_ema=None, ema_momentum=0.1, eps=1e-6):
    """
    Applies standard normalization to the input tensor.
    Data can be either a 2D or 3D tensor.
    """
    ndims = len(data.shape)
    assert ndims in (2, 3), "Data must be either 2D or 3D, received: {}".format(ndims)

    all_data = concat_all_gather(data.contiguous())

    # Compute mean and std over the first dimension.
    # If data is 3D, then compute the mean and std
    # over the first two dimensions.
    dims = [0]
    if ndims == 3:
        dims.append(1)
    mean = all_data.mean(dim=dims, keepdim=True)
    std = all_data.std(dim=dims, keepdim=True) + eps

    if mean_ema is None:
        data = (data - mean) / std
    else:
        # print(mean_ema.shape, mean.shape)
        assert mean_ema.shape == mean.shape
        assert std_ema.shape == std.shape
        data = (data - mean_ema) / (std_ema + eps)
        mean_ema.copy_(mean_ema * (1 - ema_momentum) + mean * ema_momentum)
        std_ema.copy_(std_ema * (1 - ema_momentum) + std * ema_momentum)

    return data


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not distributed.is_enabled():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


import torch
from einops import rearrange


# def unpatchify(tokens: torch.Tensor, patch_size: Optional[int] = None,
#                image_size: Optional[int] = None) -> torch.Tensor:
#     """
#     Converts a sequence of ViT tokens back into a 2D feature map.
#
#     Args:
#         tokens (torch.Tensor): The token tensor of shape (B, N, D),
#                                where N is the number of patches.
#         patch_size (int, optional): The size of the patch. If provided, image_size is ignored.
#         image_size (int, optional): The size of the original image. Used if patch_size is not given.
#
#     Returns:
#         torch.Tensor: The feature map of shape (B, D, H, W).
#     """
#     if tokens.ndim != 3:
#         raise ValueError(f"Input tokens must be a 3D tensor (B, N, D), but got shape {tokens.shape}")
#
#     num_patches = tokens.shape[1]
#
#     # --- 计算特征图的高度和宽度 (H, W) ---
#     # 通常特征图是方形的
#     h = w = int(num_patches ** 0.5)
#
#     # 一个健壮性检查，确保token数量是平方数
#     if h * w != num_patches:
#         raise ValueError(f"The number of patches ({num_patches}) is not a perfect square. "
#                          "Cannot reshape into a square feature map.")
#
#     # --- 使用 einops.rearrange进行转换 ---
#     # 'b (h w) c' -> 'b c h w'
#     # b: batch size
#     # h: height of the feature map (in patches)
#     # w: width of the feature map (in patches)
#     # c: channel dimension (embedding dim)
#     feature_map = rearrange(tokens, 'b (h w) c -> b c h w', h=h, w=w)
#
#     return feature_map
