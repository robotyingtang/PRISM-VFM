import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed
from timm.scheduler.cosine_lr import CosineLRScheduler
import gc

import torch.nn.functional as F
from collections import OrderedDict
from utils import global_print
import torch.distributed as dist
class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0.0, last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch / float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


def get_optimizer_scheduler(config, model):
    """
    Get optimizer and scheduler for model
    """
    params = model.parameters()

    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(config["lr"]),
            momentum=0.9,
            weight_decay=float(config["weight_decay"]),
        )

    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    else:
        raise NotImplementedError("Invalid optimizer %s!" % config["optimizer"])

    if config["scheduler"] == "poly":
        # Operate in each iteration
        assert config["max_iters"] is not None
        scheduler = PolynomialLR(
            optimizer=optimizer,
            max_iterations=int(config["max_iters"]),
            gamma=0.9,
            min_lr=0,
        )

    elif config["scheduler"] == "cosine":
        # Operate in each epoch
        assert config["max_epochs"] is not None
        assert config["warmup_epochs"] is not None
        max_epochs = int(config["max_epochs"])
        warmup_epochs = int(config["warmup_epochs"])
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=max_epochs - warmup_epochs,
            lr_min=1.25e-6,
            warmup_t=warmup_epochs,
            warmup_lr_init=1.25e-7,
            warmup_prefix=True,
        )

    else:
        raise NotImplementedError("Invalid scheduler %s!" % config["scheduler"])

    return optimizer, scheduler


def get_optimizer_scheduler_by_param_groups(config, param_groups, lr, weight_decay):
    """
    Get optimizer and scheduler for model
    """
    params = param_groups

    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    else:
        raise NotImplementedError("Invalid optimizer %s!" % config["optimizer"])

    if config["scheduler"] == "poly":
        # Operate in each iteration
        assert config["max_iters"] is not None
        scheduler = PolynomialLR(
            optimizer=optimizer,
            max_iterations=int(config["max_iters"]),
            gamma=0.9,
            min_lr=0,
        )

    elif config["scheduler"] == "cosine":
        # Operate in each epoch
        assert config["max_epochs"] is not None
        assert config["warmup_epochs"] is not None
        max_epochs = int(config["max_epochs"])
        warmup_epochs = int(config["warmup_epochs"])
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=max_epochs - warmup_epochs,
            lr_min=1.25e-6,
            warmup_t=warmup_epochs,
            warmup_lr_init=1.25e-7,
            warmup_prefix=True,
        )

    else:
        raise NotImplementedError("Invalid scheduler %s!" % config["scheduler"])

    return optimizer, scheduler


def cal_params(model):
    tea_params = 0
    tea_trainable_params = 0
    stu_params = 0
    stu_trainable_params = 0
    stu_vit_params = 0
    stu_vfm_projection_params = 0
    # stu_aligner_params = 0
    stu_moe_params = 0
    stu_head_params = 0
    stu_condition_embeddings_params = 0


    for name, param in model.named_parameters():
        if "teacher" in name:
            tea_params += param.numel()
            if param.requires_grad:
                tea_trainable_params += param.numel()
        else:
            stu_params += param.numel()
            if param.requires_grad:
                stu_trainable_params += param.numel()

            if "backbone" in name:
                stu_vit_params += param.numel()
            elif "vfm_projection_heads" in name:
                stu_vfm_projection_params += param.numel()
            # elif "ts_aligners" in name:
            #     stu_aligner_params += param.numel()
            elif "moe_ffn_layer" in name:
                stu_moe_params += param.numel()
            elif "head" in name:
                stu_head_params += param.numel()
            elif "condition_embeddings" in name:
                stu_condition_embeddings_params += param.numel()

    # Print a table
    print("--- Number of parameters ---")
    print(f"Teachers:     {tea_params/1e6:>10.2f}M")
    print(f"Trainable:    {tea_trainable_params/1e6:>10.2f}M")
    print(f"Student:      {stu_params/1e6:>10.2f}M")
    print(f"Trainable:    {stu_trainable_params/1e6:>10.2f}M")
    print(f"ViT:          {stu_vit_params/1e6:>10.2f}M")
    print(f"VfM Proj:     {stu_vfm_projection_params / 1e6:>10.2f}M")
    # print(f"Aligners:     {stu_aligner_params/1e6:>10.2f}M")
    print(f"MoE:          {stu_moe_params/1e6:>10.2f}M")
    print(f"Heads:        {stu_head_params/1e6:>10.2f}M")
    print(f"Cond Embedd:  {stu_condition_embeddings_params/1e6:>10.2f}M")

    return stu_params

import gc
import torch

def _resolve_pos_embed_owner(model):
    """
    返回真正持有 patch_embed / pos_embed 的 ViT 模块。
    兼容：
    1) model.patch_embed
    2) model.backbone.patch_embed
    3) model.vit.patch_embed   <- PRISM
    4) model.backbone.vit.patch_embed
    """
    candidates = []

    candidates.append(model)

    if hasattr(model, "backbone"):
        candidates.append(model.backbone)

    if hasattr(model, "vit"):
        candidates.append(model.vit)

    if hasattr(model, "backbone") and hasattr(model.backbone, "vit"):
        candidates.append(model.backbone.vit)

    for m in candidates:
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "grid_size"):
            return m

    raise AttributeError(
        "Cannot find a module that owns patch_embed.grid_size. "
        "Tried model / model.backbone / model.vit / model.backbone.vit."
    )


# def update_weights(model, state_dict):
#     """
#     更健壮的权重加载函数。
#     兼容 Condition_MoE_PRISM / 其他 ViT 包装结构。
#     """
#     new_state_dict = model.state_dict()
#     matched_keys = []
#
#     pos_owner = _resolve_pos_embed_owner(model)
#     # global_print("====================================keys in model====================================")
#     # for k, v in new_state_dict.items():
#     #     global_print(f"'{k}',")
#     # global_print("====================================keys in state====================================")
#     # for k, v in state_dict.items():
#     #     global_print(f"'{k}',")
#
#     for k, v in new_state_dict.items():
#         global_print(f"Processing model key '{k}'...")
#         old_k = None
#
#         if k in state_dict:
#             print(f"Mapping key '{k}' directly from state_dict.")
#             old_k = k
#         elif k.replace("backbone.", "") in state_dict:
#             mapped_k = k.replace("backbone.", "")
#             print(f"Mapping key '{k}' to '{mapped_k}' from state_dict.")
#             old_k = mapped_k
#         elif k.replace("vit.", "") in state_dict:
#             mapped_k = k.replace("vit.", "")
#             print(f"Mapping key '{k}' to '{mapped_k}' from state_dict.")
#             old_k = mapped_k
#         elif ("backbone." + k) in state_dict:
#             mapped_k = "backbone." + k
#             print(f"Mapping key '{k}' to '{mapped_k}' from state_dict.")
#             old_k = mapped_k
#         elif ("vit." + k) in state_dict:
#             mapped_k = "vit." + k
#             print(f"Mapping key '{k}' to '{mapped_k}' from state_dict.")
#             old_k = mapped_k
#         else:
#             print(f"No matching key found for '{k}' in state_dict.")
#             continue
#
#         # 位置编码特殊处理
#         if "pos_embed" in k:
#             pretrain_v = state_dict[old_k]
#             if pretrain_v.shape != v.shape:
#                 print(f"Resampling pos_embed for key '{k}' from {pretrain_v.shape} to {v.shape}")
#                 pretrain_v = resample_abs_pos_embed(
#                     pretrain_v,
#                     new_size=pos_owner.patch_embed.grid_size,
#                     num_prefix_tokens=1,
#                     interpolation="bicubic",
#                     antialias=False,
#                     verbose=True,
#                 )
#             new_state_dict[k] = pretrain_v
#             matched_keys.append(k)
#
#         # task condition embedding 一般保留初始化
#         elif "task_condition_embeddings" in k:
#             print(f"Skipping '{k}' to retain its initial weights for new tasks.")
#             continue
#
#         else:
#             if v.shape == state_dict[old_k].shape:
#                 new_state_dict[k] = state_dict[old_k]
#                 matched_keys.append(k)
#             else:
#                 print(
#                     f"Skipping '{k}' due to shape mismatch: "
#                     f"model shape {v.shape}, pretrain shape {state_dict[old_k].shape}"
#                 )
#
#     missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
#
#     print(f"Loaded {len(matched_keys)} weights from the provided state_dict.")
#     if missing_keys:
#         print("Missing keys in state_dict that were not loaded:", missing_keys)
#     if unexpected_keys:
#         print("Unexpected keys in state_dict that were not used:", unexpected_keys)
#
#     del new_state_dict
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()

def update_weights(model, state_dict):
    new_state_dict = model.state_dict()
    matched_keys = []

    pos_owner = _resolve_pos_embed_owner(model)

    for k, v in new_state_dict.items():
        global_print(f"Processing model key '{k}'...")

        # 这些一般保留初始化
        if "task_condition_embeddings" in k:
            print(f"Skipping '{k}' to retain its initial weights for new tasks.")
            continue

        old_k = None

        # 1. exact match
        if k in state_dict:
            old_k = k

        # 2. backbone.xxx -> xxx
        elif k.startswith("backbone.") and k[len("backbone."):] in state_dict:
            old_k = k[len("backbone."):]

        # 3. vit.xxx -> xxx
        elif k.startswith("vit.") and k[len("vit."):] in state_dict:
            old_k = k[len("vit."):]

        # 4. xxx -> backbone.xxx
        elif ("backbone." + k) in state_dict:
            old_k = "backbone." + k

        # 5. xxx -> vit.xxx
        elif ("vit." + k) in state_dict:
            old_k = "vit." + k

        # 6. backbone.xxx -> vit.xxx   <-- 你现在缺的就是这个
        elif k.startswith("backbone.") and ("vit." + k[len("backbone."):]) in state_dict:
            old_k = "vit." + k[len("backbone."):]

        # 7. vit.xxx -> backbone.xxx
        elif k.startswith("vit.") and ("backbone." + k[len("vit."):]) in state_dict:
            old_k = "backbone." + k[len("vit."):]

        else:
            print(f"No matching key found for '{k}' in state_dict.")
            continue

        print(f"Mapping key '{k}' to '{old_k}' from state_dict.")

        if "pos_embed" in k:
            pretrain_v = state_dict[old_k]
            if pretrain_v.shape != v.shape:
                print(f"Resampling pos_embed for key '{k}' from {pretrain_v.shape} to {v.shape}")
                pretrain_v = resample_abs_pos_embed(
                    pretrain_v,
                    new_size=pos_owner.patch_embed.grid_size,
                    num_prefix_tokens=1,
                    interpolation="bicubic",
                    antialias=False,
                    verbose=True,
                )
            new_state_dict[k] = pretrain_v
            matched_keys.append(k)

        else:
            if v.shape == state_dict[old_k].shape:
                new_state_dict[k] = state_dict[old_k]
                matched_keys.append(k)
            else:
                print(
                    f"Skipping '{k}' due to shape mismatch: "
                    f"model shape {v.shape}, pretrain shape {state_dict[old_k].shape}"
                )

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(matched_keys)} weights from the provided state_dict.")
    if missing_keys:
        print("Missing keys in model after loading:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys in state_dict that were not used:", unexpected_keys)

    del new_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def update_s2_weights_backup(model, state_dict):
    """
    更健壮的权重加载函数。

    :param model: 要加载权重的目标模型。
    :param state_dict: 包含预训练权重的状态字典。
    """
    # 1. 创建一个新的 state_dict 用于加载，只包含在预训练权重中存在的键
    new_state_dict = model.state_dict()
    matched_keys = []

    for k, v in new_state_dict.items():
        # 尝试匹配预训练 state_dict 中的键
        if "task_composer_weights" in k:
            print(v)

        old_k = None
        if k in state_dict:
            print(f"Mapping key '{k}' directly from state_dict.")
            old_k = k
        elif k.replace("backbone.", "") in state_dict:
            print(f"Mapping key '{k}' to '{k.replace('backbone.', '')}' from state_dict.")
            # 处理当模型将 ViT 包装在 "backbone" 模块下的情况
            old_k = k.replace("backbone.", "")
        elif "backbone." + k in state_dict:

            old_k = "backbone." + k
            print(f"Mapping key '{k}' to '{old_k}' from state_dict.")
        # 可以根据需要添加更多的键名匹配规则，例如处理 LoRA 权重的键
        # elif "lora" in k and ...:
        #     ...

        if old_k is None:
            # 如果在 state_dict 中找不到匹配的键，则跳过，保留模型的初始权重
            continue

        # 2. 特殊处理位置编码 (pos_embed)
        if "pos_embed" in k:
            pretrain_v = state_dict[old_k]
            # 如果尺寸不匹配，进行重采样
            if pretrain_v.shape != v.shape:
                print(f"Resampling pos_embed for key '{k}' from {pretrain_v.shape} to {v.shape}")
                # 确定 backbone 对象
                backbone = model.backbone if hasattr(model, "backbone") else model
                pretrain_v = resample_abs_pos_embed(
                    pretrain_v,
                    new_size=backbone.patch_embed.grid_size,
                    num_prefix_tokens=1, # 假设总是有 1 个 class token
                    interpolation="bicubic",
                    antialias=False,
                    verbose=True,
                )
            new_state_dict[k] = pretrain_v
            matched_keys.append(k)
        # 3. 特殊处理 task_condition_embeddings
        elif "task_condition_embeddings" in k:
            # 这里的逻辑需要根据你的意图来确定。
            # 如果预训练模型有这个权重且你希望加载它，可以取消下面的注释。
            if old_k in state_dict:
               new_state_dict[k] = state_dict[old_k]
               matched_keys.append(k)
            # 否则，跳过这个键，让它保留随机初始化，这在微调新任务时是常见的做法。
            else:
                print(f"Skipping '{k}' to retain its initial weights for new tasks.")
            continue
        # 4. 复制其他所有匹配的权重
        else:
            if v.shape == state_dict[old_k].shape:
                new_state_dict[k] = state_dict[old_k]
                matched_keys.append(k)
            else:
                print(f"Skipping '{k}' due to shape mismatch: model shape {v.shape}, pretrain shape {state_dict[old_k].shape}")


    # 5. 使用最终构建的 state_dict 加载模型
    # strict=False 仍然是推荐的，因为它会忽略 new_state_dict 中没有的键（例如我们主动跳过的那些）
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(matched_keys)} weights from the provided state_dict.")
    if missing_keys:
        print("Missing keys in state_dict that were not loaded:", missing_keys)
    if unexpected_keys:
        # 在这种逻辑下，unexpected_keys 应该为空
        print("Unexpected keys in state_dict that were not used:", unexpected_keys)


    # 6. 清理内存
    del new_state_dict
    torch.cuda.empty_cache()
    gc.collect()
# def update_weights(model, state_dict):
#     # print(model.state_dict())
#     # print("##############################################################################################")
#     # print(state_dict)
#     model.load_state_dict(state_dict, strict=False)
#     return
#
#     update_state_dict = model.state_dict()
#
#     for k in update_state_dict.keys():
#         # kk = k.replace("module.", "")  # model is wrapped by DDP
#         if k in state_dict:  # same arch
#             old_k = k
#         elif k.replace("backbone.", "") in state_dict:  # distilled in backbone, now multi-task model
#             old_k = k.replace("backbone.", "")
#         else:
#             continue
#
#         if "task_condition_embeddings" in k:
#             print(k)
#
#             # update_state_dict[k] = state_dict["task_free_condition_embeddings.weight"].expand( update_state_dict[k].shape[0], -1, )
#             update_state_dict[k] = state_dict["backbone.task_condition_embeddings.weight"]
#             # Skip task condition embeddings, which are not in the state_dict
#             continue
#         if "pos_embed" in k:
#             v = state_dict[old_k]
#             # # Resize pos embedding
#             if v.shape[1] != update_state_dict[k].shape[1]:
#                 if hasattr(model, "backbone"):
#                     backbone = model.backbone
#                 else:
#                     backbone = model
#                 v = resample_abs_pos_embed(
#                     v,
#                     new_size=backbone.patch_embed.grid_size,
#                     num_prefix_tokens=1,
#                     interpolation="bicubic",
#                     antialias=False,
#                     verbose=True,
#                 )
#             update_state_dict[k] = v
#         else:
#             update_state_dict[k] = state_dict[old_k]
#
#     model.load_state_dict(update_state_dict,strict=False)
#
#
#     del update_state_dict
#     torch.cuda.empty_cache()
#     gc.collect()
    # print("############################################################################")
    # print(model.state_dict())


# def resample_abs_pos_embed(
#         posemb,
#         new_size,
#         num_prefix_tokens=1,
#         interpolation='bicubic',
#         antialias=False,
#         verbose=False):
#     """
#     Resample absolute position embedding to a new resolution.
#     """
#     if posemb.shape[1] - num_prefix_tokens == new_size[0] * new_size[1]:
#         return posemb
#
#     # 获取 class token 或其他 prefix tokens
#     prefix_posemb = posemb[:, :num_prefix_tokens]
#     # 获取 grid position embedding
#     grid_posemb = posemb[:, num_prefix_tokens:]
#
#     # 计算旧的 grid size
#     gs_old_h = int(torch.sqrt(torch.tensor(grid_posemb.shape[1])).item())
#     gs_old_w = gs_old_h
#
#     if verbose:
#         print(
#             f'Resampling pos_embed from {posemb.shape} to grid size {new_size} with {num_prefix_tokens} prefix tokens.')
#
#     # 将 1D 的 grid embedding 重塑为 2D
#     grid_posemb = grid_posemb.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
#
#     # 使用 F.interpolate 进行重采样
#     grid_posemb = F.interpolate(
#         grid_posemb,
#         size=new_size,
#         mode=interpolation,
#         antialias=antialias,
#         align_corners=False,
#     )
#
#     # 将 2D grid 重塑回 1D 并与 prefix token 拼接
#     grid_posemb = grid_posemb.permute(0, 2, 3, 1).reshape(1, new_size[0] * new_size[1], -1)
#     posemb = torch.cat([prefix_posemb, grid_posemb], dim=1)
#
#     return posemb


def update_s2_weights(model, state_dict, verbose=True):
    """
    最终版权重加载函数，能精确处理多种前缀不匹配问题。

    :param model: 要加载权重的目标模型。
    :param state_dict: 包含预训练权重的状态字典。
    :param verbose: 是否打印详细的加载日志。
    """
    weights_to_load = OrderedDict()
    model_state_dict = model.state_dict()
    checkpoint_keys = set(state_dict.keys())

    loaded_keys = set()
    used_checkpoint_keys = set()

    global_print("====================================keys in model====================================")
    for model_key, model_tensor in model_state_dict.items():
        global_print(model_key)
    global_print("=======================================================================================")
    global_print("=======================================================================================")
    global_print("====================================keys in checkpoint==================================")
    for ckpt_key in state_dict.keys():
        global_print(ckpt_key)
    global_print("=======================================================================================")
    # 1. 遍历模型的所有参数，尝试在 checkpoint 中找到匹配项
    for model_key, model_tensor in model_state_dict.items():

        # 核心改进：生成一个潜在匹配键的有序列表，从最可能到最不可能
        potential_ckpt_keys = []

        # 规则 1: 直接匹配
        potential_ckpt_keys.append(model_key)

        # 规则 2: 处理双重 backbone 嵌套 -> 单 backbone (最常见的问题)
        if model_key.startswith('backbone.backbone.'):
            # 将 'backbone.backbone.' 替换为 'backbone.'
            potential_ckpt_keys.append(model_key.replace('backbone.backbone.', 'backbone.', 1))

        # 规则 3: 处理模型有 backbone 而 checkpoint 没有的情况
        if model_key.startswith('backbone.'):
             # 移除开头的 'backbone.'
            potential_ckpt_keys.append(model_key.replace('backbone.', '', 1))

        # 规则 4: 处理 DataParallel/DDP 的 'module.' 前缀
        if model_key.startswith('module.'):
            potential_ckpt_keys.append(model_key.replace('module.', '', 1))

        # 也可以添加反向规则，例如模型没有 module. 而 checkpoint 有
        potential_ckpt_keys.append(f'module.{model_key}')

        # 去重并保持顺序
        potential_ckpt_keys = list(OrderedDict.fromkeys(potential_ckpt_keys))

        ckpt_key_found = None
        for key in potential_ckpt_keys:
            if key in state_dict:
                ckpt_key_found = key
                break

        if ckpt_key_found is None:
            continue

        ckpt_tensor = state_dict[ckpt_key_found]

        # 2. 特殊逻辑处理
        if "task_condition_embeddings" in model_key:

            if verbose: global_print(f"INFO: Skipping '{model_key}' for new tasks.")
            continue

        if model_tensor.shape != ckpt_tensor.shape:
            if "pos_embed" in model_key:
                if verbose: global_print(f"INFO: Resampling '{model_key}' from {ckpt_tensor.shape} to {model_tensor.shape}.")
                try:
                    backbone = model.backbone.backbone # 假设是双重嵌套
                    ckpt_tensor = resample_abs_pos_embed(
                        ckpt_tensor,
                        new_size=backbone.patch_embed.grid_size,
                        num_prefix_tokens=1
                    )
                except Exception as e:
                    if verbose: global_print(f"WARNING: Failed to resample '{model_key}': {e}. Skipping.")
                    continue
            else:
                if verbose: global_print(f"WARNING: Skipping '{model_key}' due to shape mismatch: model={model_tensor.shape}, ckpt={ckpt_tensor.shape}.")
                continue

        weights_to_load[model_key] = ckpt_tensor
        loaded_keys.add(model_key)
        used_checkpoint_keys.add(ckpt_key_found)
        if verbose and ckpt_key_found != model_key:
            global_print(f"INFO: Mapped model key '{model_key}' from checkpoint key '{ckpt_key_found}'.")

    # 3. 执行加载
    missing_keys, unexpected_keys = model.load_state_dict(weights_to_load, strict=False)

    # 4. 生成最终报告
    global_print("\n" + "="*50)
    global_print("         Weight Loading Report (V2)")
    global_print("="*50)
    global_print(f"Successfully loaded {len(loaded_keys)} weights into the model.")

    unloaded_model_keys = set(model_state_dict.keys()) - loaded_keys
    if unloaded_model_keys:
        global_print(f"\n--- {len(unloaded_model_keys)} keys in the model were NOT loaded ---")
        # 打印部分未加载的键作为示例
        for i, key in enumerate(list(unloaded_model_keys)):
            global_print(f"  - {key}")
        # if len(unloaded_model_keys) > 100:
        #     global_print("  - ... (and others)")

    unused_ckpt_keys = checkpoint_keys - used_checkpoint_keys
    if unused_ckpt_keys:
        global_print(f"\n--- {len(unused_ckpt_keys)} keys in the checkpoint were NOT used ---")
        for i, key in enumerate(list(unused_ckpt_keys)):
            global_print(f"  - {key}")
        # if len(unused_ckpt_keys) > 100:
        #     global_print("  - ... (and others)")

    # `unexpected_keys` from load_state_dict should be empty with our logic, but we check just in case.
    if unexpected_keys:
        global_print(f"\n--- {len(unexpected_keys)} unexpected keys were found during loading ---")
        global_print("This is unusual with this script. Please check:", unexpected_keys)

    global_print("="*50 + "\n")

def freeze_experts_by_indices(model: nn.Module, expert_indices_to_freeze: list, freeze_shared_expert: bool = False):
    """
    遍历模型，找到所有的 ConditionedMoELayer，并冻结指定索引的专家。

    Args:
        model (nn.Module): 你的整个模型实例 (e.g., Condition_MoE_PRISM).
        expert_indices_to_freeze (list): 一个包含要冻结的专家索引的列表, e.g., [0, 1, 2, 3].
    """
    print(f"Attempting to freeze experts at indices: {expert_indices_to_freeze}")

    total_frozen_params = 0
    experts_found = False

    for module in model.modules():
        # 找到我们自定义的MoE FFN层
        if hasattr(module, 'moe_ffn_layer') and hasattr(module.moe_ffn_layer, 'experts'):
            experts_found = True
            moe_layer = module.moe_ffn_layer

            for i, expert in enumerate(moe_layer.experts):

                if i in expert_indices_to_freeze:
                    for param in expert.parameters():
                        if param.requires_grad:
                            param.requires_grad = False
                            total_frozen_params += param.numel()
                    # print(f"  - Froze expert {i} in a ConditionedMoELayer.")

            if freeze_shared_expert and hasattr(moe_layer, 'shared_expert'):
                for param in moe_layer.shared_expert.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        total_frozen_params += param.numel()

    if not experts_found:
        print("Warning: No 'moe_ffn_layer' with an 'experts' attribute found in the model.")
    else:
        print(f"Successfully froze {total_frozen_params / 1e6:.2f}M parameters from specified experts.")

def create_s2_param_groups_with_logging(model, base_lr, weight_decay, new_module_lr_multiplier=10.0, head_lr_multiplier=10.0):
    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # 定义参数组
    param_groups = {
        'pretrained': {'params': [], 'lr': base_lr, 'weight_decay': weight_decay},
        'new_modules': {'params': [], 'lr': base_lr * new_module_lr_multiplier, 'weight_decay': weight_decay},
        'heads': {'params': [], 'lr': base_lr * head_lr_multiplier, 'weight_decay': weight_decay}
    }

    # 定义名称匹配规则
    # ！！！请根据你模型中模块的实际命名进行调整！！！
    new_module_keywords = ["task_gate", "task_floor_logits", "gate_linear"] #,'film_layer','W_g_base','W_noise'
    head_keywords = []  #'.heads' 假设你的任务头和VFM投影头都在这里

    # 2. 初始化 6 个参数桶 (3个模块类别 * 2种衰减策略)
    # 结构: groups[模块名][是否衰减] = list(params)


    groups = {
        'pretrained': {'decay': [], 'no_decay': [], 'lr': base_lr},
        'new_modules': {'decay': [], 'no_decay': [], 'lr': base_lr * new_module_lr_multiplier},
        'heads': {'decay': [], 'no_decay': [], 'lr': base_lr}
    }

    # 3. 遍历所有参数
    for name, param in model_to_inspect.named_parameters():
        if not param.requires_grad:
            continue

        global_print(name)
        # --- 步骤 A: 确定属于哪个模块 (pretrained, new_modules, 或 heads) ---
        group_name = 'pretrained'  # 默认

        # 检查是否属于任务头
        for keyword in head_keywords:
            if keyword in name:
                group_name = 'heads'
                break

        # 如果不是头，检查是否属于新增模块
        if group_name == 'pretrained':
            for keyword in new_module_keywords:
                if keyword in name:
                    group_name = 'new_modules'
                    break

        if group_name == 'pretrained':
            if param.ndim <= 1 or name.endswith(".bias"):
                groups[group_name]['no_decay'].append(param)
            else:
                groups[group_name]['decay'].append(param)

            # 策略 B: 对于 New Modules 和 Heads (10x LR)，强制全部 Decay
            # 即使是 Bias 和 Norm 也要 Decay，防止在高 LR 下漂移
        elif group_name == 'new_modules':
            groups[group_name]['no_decay'].append(param)
        else:
            # 这里一律放入 decay 组，除非你极其确定某个参数不能 decay
            groups[group_name]['decay'].append(param)

    # 4. 构建最终给 Optimizer 的 param_groups 列表
    final_param_groups = []

    # 用于日志统计
    log_stats = []

    for key, data in groups.items():
        lr = data['lr']

        # 添加需要衰减的组
        if len(data['decay']) > 0:
            final_param_groups.append({
                'params': data['decay'],
                'weight_decay': weight_decay,
                'lr': lr,
                'name': f"{key}_decay"
            })

        # 添加不需要衰减的组 (weight_decay = 0.0)
        if len(data['no_decay']) > 0:
            final_param_groups.append({
                'params': data['no_decay'],
                'weight_decay': 0.0,
                'lr': lr,
                'name': f"{key}_no_decay"
            })

    # --- 日志输出 ---
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("=" * 80)
        print("Optimizer Parameter Groups Configuration (With Decay Handling):")
        print("=" * 80)
        total_params = 0

        for group in final_param_groups:
            count = sum(p.numel() for p in group['params'])
            total_params += count

            g_name = group.get('name', 'unknown')
            g_lr = group['lr']
            g_wd = group['weight_decay']

            print(f"  - Group: '{g_name}'")
            print(f"    - Params: {count:,}")
            print(f"    - LR: {g_lr:.2e}, WD: {g_wd}")

        print(f"\nTotal Trainable Parameters: {total_params:,}")
        print("=" * 80)

    return final_param_groups

def create_s2_param_groups_full_decay(model, base_lr, weight_decay, new_module_lr_multiplier=10.0, head_lr_multiplier=10.0):
    """
    【对比实验组】全量衰减策略 (Full Weight Decay)
    不管参数是 Bias 还是 LayerNorm，只要它需要梯度，就施加 Weight Decay。
    这在高 LR (10x) 场景下通常比 No Decay 更稳定。
    """
    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # 1. 定义名称匹配规则 (与之前保持一致)
    # 包含了 Router, Expert, FiLM, Scale 等 MoE 相关组件
    new_module_keywords = [
        'task_condition_embeddings', 'film_layer', 'W_g_base', 'W_noise',
        'router', 'expert', 'output_scale', 'adapter'
    ]
    head_keywords = ['.heads', 'proj_']

    # 2. 初始化参数桶
    # 这里不需要 'no_decay' 桶，因为所有参数都 Decay
    groups = {
        'pretrained': {'params': [], 'lr': base_lr},
        'new_modules': {'params': [], 'lr': base_lr * new_module_lr_multiplier},
        'heads': {'params': [], 'lr': base_lr * head_lr_multiplier}
    }

    # 3. 遍历所有参数
    for name, param in model_to_inspect.named_parameters():
        if not param.requires_grad:
            continue

        # --- 步骤 A: 确定属于哪个模块 ---
        group_name = 'pretrained'  # 默认

        # 检查是否属于任务头
        for keyword in head_keywords:
            if keyword in name:
                group_name = 'heads'
                break

        # 如果不是头，检查是否属于新增模块
        if group_name == 'pretrained':
            for keyword in new_module_keywords:
                if keyword in name:
                    group_name = 'new_modules'
                    break

        # --- 步骤 B: 无论是什么参数，直接加入列表 (Full Decay) ---
        groups[group_name]['params'].append(param)

    # 4. 构建最终给 Optimizer 的 param_groups 列表
    final_param_groups = []

    for key, data in groups.items():
        if len(data['params']) > 0:
            final_param_groups.append({
                'params': data['params'],
                'weight_decay': weight_decay, # 统一使用传入的 weight_decay
                'lr': data['lr'],
                'name': f"{key}_full_decay"
            })

    # --- 日志输出 ---
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("=" * 80)
        print("Optimizer Parameter Groups Configuration (FULL DECAY MODE):")
        print(f"Warning: Bias and LayerNorm weights will also be decayed.")
        print("=" * 80)
        total_params = 0

        for group in final_param_groups:
            count = sum(p.numel() for p in group['params'])
            total_params += count

            g_name = group.get('name', 'unknown')
            g_lr = group['lr']
            g_wd = group['weight_decay']

            print(f"  - Group: '{g_name}'")
            print(f"    - Params: {count:,}")
            print(f"    - LR: {g_lr:.2e}, WD: {g_wd}")

        print(f"\nTotal Trainable Parameters: {total_params:,}")
        print("=" * 80)

    return final_param_groups


def create_s1_param_groups_with_logging(model, base_lr, weight_decay, new_module_lr_multiplier=10.0, head_lr_multiplier=10.0):
    """
    为模型创建差分学习率的参数组，并提供详细的日志输出。

    Args:
        model (nn.Module): 你的DDP模型（或原始模型）。
        base_lr (float): 预训练部分的基础学习率。
        weight_decay (float): 权重衰减值。
        new_module_lr_multiplier (float): 新增模块（MoE, Gate）相对于base_lr的乘数。
        head_lr_multiplier (float): 任务头相对于base_lr的乘数。

    Returns:
        list: 一个适用于torch.optim.AdamW的参数组列表。
    """
    # 如果是DDP模型，我们需要访问 .module
    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # # 定义参数组
    # param_groups = {
    #     'pretrained': {'params': [], 'lr': base_lr, 'weight_decay': weight_decay},
    #     'new_modules': {'params': [], 'lr': base_lr * new_module_lr_multiplier, 'weight_decay': weight_decay},
    #     'heads': {'params': [], 'lr': base_lr * head_lr_multiplier, 'weight_decay': weight_decay}
    # }

    # 定义名称匹配规则
    # ！！！请根据你模型中模块的实际命名进行调整！！！
    new_module_keywords = ['moe_ffn_layer', 'gate','vfm_condition_embeddings','vfm_free_embeddings','task_condition_embeddings']
    head_keywords = ['vfm_projection_heads']  # 假设你的任务头和VFM投影头都在这里

    # 2. 初始化 6 个参数桶 (3个模块类别 * 2种衰减策略)
    # 结构: groups[模块名][是否衰减] = list(params)
    groups = {
        'pretrained': {'decay': [], 'no_decay': [], 'lr': base_lr},
        'new_modules': {'decay': [], 'no_decay': [], 'lr': base_lr * new_module_lr_multiplier},
        'heads': {'decay': [], 'no_decay': [], 'lr': base_lr * head_lr_multiplier}
    }

    # 3. 遍历所有参数
    for name, param in model_to_inspect.named_parameters():
        if not param.requires_grad:
            continue

        # --- 步骤 A: 确定属于哪个模块 (pretrained, new_modules, 或 heads) ---
        group_name = 'pretrained'  # 默认

        # 检查是否属于任务头
        for keyword in head_keywords:
            if keyword in name:
                group_name = 'heads'
                break

        # 如果不是头，检查是否属于新增模块
        if group_name == 'pretrained':
            for keyword in new_module_keywords:
                if keyword in name:
                    group_name = 'new_modules'
                    break

        # --- 步骤 B: 确定是否需要 weight_decay ---
        # 如果参数名包含 bias，或者属于 Norm 层 (通常名字里带 norm.weight)，则不衰减
        # 注意：有些 LayerNorm 的权重叫 weight，但也应该免除 decay。
        # 简单的判断方法是：ndim <= 1 的参数通常不衰减 (bias是1维, layernorm weight是1维)
        if param.ndim <= 1 or name.endswith(".bias"):
            groups[group_name]['no_decay'].append(param)
        else:
            groups[group_name]['decay'].append(param)

    # 4. 构建最终给 Optimizer 的 param_groups 列表
    final_param_groups = []

    # 用于日志统计
    log_stats = []

    for key, data in groups.items():
        lr = data['lr']

        # 添加需要衰减的组
        if len(data['decay']) > 0:
            final_param_groups.append({
                'params': data['decay'],
                'weight_decay': weight_decay,
                'lr': lr,
                'name': f"{key}_decay"
            })

        # 添加不需要衰减的组 (weight_decay = 0.0)
        if len(data['no_decay']) > 0:
            final_param_groups.append({
                'params': data['no_decay'],
                'weight_decay': 0.0,
                'lr': lr,
                'name': f"{key}_no_decay"
            })

    # --- 日志输出 ---
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("=" * 80)
        print("Optimizer Parameter Groups Configuration (With Decay Handling):")
        print("=" * 80)
        total_params = 0

        for group in final_param_groups:
            count = sum(p.numel() for p in group['params'])
            total_params += count

            g_name = group.get('name', 'unknown')
            g_lr = group['lr']
            g_wd = group['weight_decay']

            print(f"  - Group: '{g_name}'")
            print(f"    - Params: {count:,}")
            print(f"    - LR: {g_lr:.2e}, WD: {g_wd}")

        print(f"\nTotal Trainable Parameters: {total_params:,}")
        print("=" * 80)

    return final_param_groups


def reset_router_bias(model):
    print(">>> Reseting Router Bias to 0 to break the 'Winner-Take-All' loop...")

    # 遍历模型找到所有的 Router
    # 注意：根据你的模型层级结构调整 getattr 路径
    student = model.module.student if hasattr(model, 'module') else model.student

    count = 0
    for name, module in student.named_modules():
        # 找到 Router 的线性层 (通常叫 W_g_base 或 fc)
        if "router" in name and isinstance(module, torch.nn.Linear):
            # 1. 暴力清零 Bias
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            count += 1
            # 2. (可选但推荐) 增加 Weight 的扰动，打破僵局
            # nn.init.normal_(module.weight, std=0.01)
        if "to_gamma_beta" in name and isinstance(module, torch.nn.Linear):
            nn.init.constant_(module.weight, 0.0)
            nn.init.constant_(module.bias, 0.0)


    print(f">>> Reset {count} routers. Now they listen to the input!")
