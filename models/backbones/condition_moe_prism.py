import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random

from utils import global_print
from models.projector import build_projector
from collections import OrderedDict
import copy

from functools import partial
from models.backbones.vit import Block


from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

import torch
import torch.nn as nn
from einops import rearrange
from timm.data import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.layers import (
    PatchEmbed,
    Mlp,
    DropPath,
    PatchDropout,
    trunc_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    use_fused_attn,
    get_act_layer,
    get_norm_layer,
    LayerType,
)
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, adapt_input_conv
from models.backbones.moh_attention import MoH_ViT
from models.backbones.router import ConditionedRouter, StructureConditionedRouter, AdvancedConditionedRouter, DecoupledGroupRouter
# 确保你有一个标准的Block实现
# out_indices_cfg_for_task = {
#     "small": [2, 5, 8, 11],
#     "base": [8, 9, 10, 11], #[2, 5, 8, 11],#
#     "large": [5, 11, 17, 23],
#     "huge": [7, 15, 23, 31],
#     "giant": [9, 19, 29, 39],
#     "so": [6, 13, 20, 26],
# }

# --- 辅助模块 ---

class MlpProjector(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        # 通常中间层的维度保持与 input_dim 一致，或者通过 mlp_ratio 放大
        hidden_dim = int(input_dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)




# class ConstrainedSigmoid(nn.Module):
#     def forward(self, x):
#         # 将输出范围从 (0, 1) 映射到 (0.5, 1.0)
#         return 0.5 * torch.sigmoid(x) + 0.5


class ConstrainedSigmoid(nn.Module):
    """
    一个Sigmoid激活函数，将其输出范围从 (0, 1) 线性映射到 (min_val, max_val)。
    """

    def __init__(self, min_val: float = 0.5, max_val: float = 1.0):
        super().__init__()
        if not 0.0 <= min_val < max_val <= 1.0:
            raise ValueError(f"Range ({min_val}, {max_val}) is invalid. "
                             "Must satisfy 0.0 <= min_val < max_val <= 1.0.")

        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 预激活值。
        Returns:
            torch.Tensor: 映射到指定范围内的激活值。
        """
        # torch.sigmoid(x) -> (0, 1)
        # torch.sigmoid(x) * self.range -> (0, range)
        # torch.sigmoid(x) * self.range + self.min_val -> (min_val, min_val + range) = (min_val, max_val)
        return torch.sigmoid(x) * self.range + self.min_val

    def __repr__(self):
        return f"ConstrainedSigmoid(min_val={self.min_val}, max_val={self.max_val})"




# --- MoE 组件 ---
class UniversalExpert(nn.Module):
    """通用Expert，通常是一个FFN（前馈网络）"""

    def __init__(self, input_dim, expert_hidden_ratio, output_dim: int = None, dropout=0.0):
        # dim: 输入和输出特征维度
        # hidden_dim_ratio: FFN隐层维度相对于输入维度的比例
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        hidden_dim = int(input_dim * expert_hidden_ratio)  # 计算隐层维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()  # GELU激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x




class SpatialExpert(nn.Module):
    """
    一个带有空间感受野的专家，专门用于捕捉高频几何信息（Edge/Normal）。
    """

    def __init__(self, input_dim, expert_hidden_ratio, fea_size=(24, 24)):
        super().__init__()
        hidden_dim = int(input_dim * expert_hidden_ratio)
        self.fea_size = fea_size  # 例如 (384//16, 384//16) = (24, 24)

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 核心：3x3 Depthwise Conv 提取局部梯度，不改变通道间信息，参数极少
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 注意：如果是在 Dense 阶段（Stage 1 蒸馏，或者没加 Top-K 丢弃的 Token）
        # x shape: (B, N, D)
        B, N, D = x.shape
        H, W = self.fea_size

        # 如果有 CLS Token，先分离
        num_prefix = N - H * W
        prefix_tokens = x[:, :num_prefix, :]
        spatial_tokens = x[:, num_prefix:, :]

        # 1. 升维
        spatial_tokens = self.fc1(spatial_tokens)

        # 2. Reshape 成图像并进行卷积
        spatial_tokens = rearrange(spatial_tokens, 'b (h w) d -> b d h w', h=H, w=W)
        spatial_tokens = self.dwconv(spatial_tokens)
        spatial_tokens = rearrange(spatial_tokens, 'b d h w -> b (h w) d')

        # 3. 激活与降维
        spatial_tokens = self.act(spatial_tokens)
        spatial_tokens = self.fc2(spatial_tokens)

        # 处理 prefix tokens
        if num_prefix > 0:
            prefix_tokens = self.fc2(self.act(self.fc1(prefix_tokens)))
            out = torch.cat([prefix_tokens, spatial_tokens], dim=1)
        else:
            out = spatial_tokens

        return out


class SpatialSharedExpert(nn.Module):
    """
    带有空间感受野的共享专家。
    负责为所有的 Task 提取基础的局部高频几何特征（边缘/法向）。
    """

    def __init__(self, input_dim, expert_hidden_ratio, output_dim=None, dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        hidden_dim = int(input_dim * expert_hidden_ratio)

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 核心：3x3 深度可分离卷积 (Depthwise Conv)
        # groups=hidden_dim 保证了它计算极快，且只做空间聚合，不做通道混合
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, N, D)
        B, N, D = x.shape

        # 1. 第一层 Linear 升维
        x = self.fc1(x)  # (B, N, hidden_dim)

        # 2. 分离 CLS Token 和 Patch Tokens
        # 假设 ViT 只有 1 个 CLS Token (N = H*W + 1)
        # 如果你的模型有 Register Tokens，这里的 1 需要改成 1 + num_reg_tokens
        num_prefix = 1
        if N % 2 == 0 and math.isqrt(N) ** 2 == N:
            # 如果 N 正好是完全平方数 (比如 14x14=196)，说明没有 CLS token
            num_prefix = 0

        if num_prefix > 0:
            prefix_tokens = x[:, :num_prefix, :]
            patch_tokens = x[:, num_prefix:, :]
        else:
            patch_tokens = x

        # 3. 将 Patch Tokens Reshape 成图像网格进行卷积
        H = W = int(math.sqrt(patch_tokens.shape[1]))
        assert H * W == patch_tokens.shape[1], f"Cannot reshape {patch_tokens.shape[1]} into a square grid."

        # (B, H*W, hidden_dim) -> (B, hidden_dim, H, W)
        patch_grid = patch_tokens.transpose(1, 2).view(B, -1, H, W)

        # 提取空间特征
        patch_grid = self.dwconv(patch_grid)

        # (B, hidden_dim, H, W) -> (B, H*W, hidden_dim)
        patch_tokens = patch_grid.view(B, -1, H * W).transpose(1, 2)

        # 合并回原始序列
        if num_prefix > 0:
            x = torch.cat([prefix_tokens, patch_tokens], dim=1)
        else:
            x = patch_tokens

        # 4. 激活与降维
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

# class UniversalExpert(nn.Module):
#     """
#     通用Expert (Enhanced Version)
#     包含 Output Norm 和 Learnable Scale 以适配多模态蒸馏
#     """
#
#     def __init__(self, input_dim, expert_hidden_ratio, output_dim: int = None, dropout=0.0, use_output_norm=True):
#         super().__init__()
#         if output_dim is None:
#             output_dim = input_dim
#
#         hidden_dim = int(input_dim * expert_hidden_ratio)
#
#         # 1. 核心网络 (保持不变)
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         # 2. 【关键改进】Output LayerNorm
#         # 作用：在Phase 2冻结Linear层时，这个LayerNorm依然可以微调(Requires Grad)，
#         # 帮助CLIP专家调整输出分布，使其能与DINO专家“平起平坐”。
#         self.use_output_norm = use_output_norm
#         if use_output_norm:
#             self.output_norm = nn.LayerNorm(output_dim)
#
#         # 3. 【关键改进】Learnable Output Scale
#         # 作用：初始化为1，允许模型自动放大(如 x2.0)或缩小专家输出。
#         # 解决了 "CLIP特征数值太小被淹没" 的问题。
#         self.output_scale = nn.Parameter(torch.ones(1))
#
#     def forward(self, x):
#         # 标准 FFN 前向传播
#         residual = x  # 如果expert包含residual连接通常在外部做，这里假设expert只是变换
#
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#
#         # 应用 LayerNorm (如果启用)
#         if self.use_output_norm:
#             x = self.output_norm(x)
#
#         # 应用可学习缩放
#         x = x * self.output_scale
#
#         return x

class ContextAwareGate(nn.Module):
    """
    一个上下文感知的门控网络。
    它结合了每个token的局部特征和来自所有token的全局上下文信息
    来做出更智能的门控决策。
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        self.dim = dim
        global_print("################ Using ContextAwareGate! ##################")


        hidden_dim = int(dim * mlp_ratio)

        # 1. 线性层处理局部token特征
        self.local_proj = nn.Linear(dim, hidden_dim)

        # 2. 线性层处理全局上下文特征
        self.global_proj = nn.Linear(dim, hidden_dim)

        # 3. 融合后的MLP
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 4. 最终的激活函数 (我们仍然保留约束，但可以放松它)
        # 范围 [0.2, 1.0]，给予MoE更大的空间，但仍防止FFN完全关闭
        self.activation = lambda x: 0.5 * torch.sigmoid(x) + 0.5

        with torch.no_grad():
            self.mlp[1].bias.fill_(2.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D)

        # --- 计算全局上下文 ---
        # 使用简单的平均池化来聚合全局信息，高效且无额外参数
        # global_context shape: (B, 1, D)
        global_context = torch.mean(x, dim=1, keepdim=True)

        # --- 结合局部与全局信息 ---
        local_feat = self.local_proj(x)  # (B, N, H)
        global_feat = self.global_proj(global_context)  # (B, 1, H) -> 广播到 (B, N, H)

        # 将局部和全局信息相加（或拼接），然后通过MLP
        # (B, N, H) + (B, N, H) -> (B, N, H)
        fused_feat = local_feat + global_feat

        # 预激活值
        pre_activation = self.mlp(fused_feat)  # (B, N, 1)

        # 最终的门控值
        g = self.activation(pre_activation)  # (B, N, 1)

        return g, pre_activation  # 同时返回预激活值，以备正则化使用


class SmarterGate(nn.Module):
    def __init__(self, dim, patch_resolution=None):
        super().__init__()
        global_print("################ Using SmarterGate! ##################")
        # self.h, self.w = patch_resolution
        # 使用一个深度可分离卷积来感知局部上下文，非常轻量
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        )
        self.activation = ConstrainedSigmoid()  # 你的约束激活函数

        with torch.no_grad():
            # self.conv[2] 访问到的是 Conv2d(dim, 1, kernel_size=1, bias=True)
            # .bias 访问它的偏置参数
            # .fill_() 原地填充我们期望的值
            self.conv[2].bias.fill_(2.0)

    def forward(self, x):
        # x: (B, N, C), N = H * W
        B, N, C = x.shape
        s = math.sqrt(N)
        # 检查 s 是否为整数 (或者非常接近整数以处理浮点误差)
        # if s != int(s):
        #     raise ValueError(f"Cannot infer a square resolution from N={N}. It is not a perfect square.")

        h = w = int(s)

        assert h * w == N, "N must be a perfect square for reshaping into a grid."
        # Reshape to 2D grid: (B, C, H, W)
        x_grid = x.transpose(1, 2).view(B, C, h, w)
        # Apply convolution
        pre_act_grid = self.conv(x_grid)  # -> (B, 1, H, W)
        # Reshape back: (B, N, 1)
        pre_act_flat = pre_act_grid.view(B, 1, N).transpose(1, 2).contiguous()

        g = self.activation(pre_act_flat)
        return g, pre_act_flat


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGateResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.05))  # 很小地起步

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, H, W):
        """
        x: [B, N, C]
        return: spatial_score [B, N, 1]
        """
        B, N, C = x.shape
        x = self.norm(x)
        x_cls = x[:, :1, :]  # [B, 1, C]
        x_patch = x[:, 1:, :]  # [B, H*W, C]

        assert x_patch.shape[1] == H * W, \
            f"x_patch tokens = {x_patch.shape[1]}, but H*W = {H * W}"

        x2d = x_patch.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        local_mean = F.avg_pool2d(x2d, kernel_size=3, stride=1, padding=1)
        contrast = torch.abs(x2d - local_mean)  # [B, C, H, W]

        contrast = contrast.flatten(2).transpose(1, 2)  # [B, H*W, C]
        spatial_score_patch = self.proj(contrast)  # [B, H*W, 1]

        # cls token 对应的 spatial score 设为 0
        spatial_score_cls = torch.zeros(
            B, 1, 1, device=x.device, dtype=x.dtype
        )

        spatial_score = torch.cat([spatial_score_cls, spatial_score_patch], dim=1)  # [B, N, 1]
        return self.alpha * spatial_score

class TaskAwareGate(nn.Module):
    """
    每个 block 一个 gate 模块。

    作用：
    1) task-specific base gate，避免所有任务学成同一个 gate
    2) optional task-specific learnable floor，避免深层永远被固定 floor=0.5 卡死
    3) token 分支只做小残差，减少训练初期抖动/坍塌

    输入:
        x:        [B, N, C]   -> ffn_input
        task_ids: [B]         -> stage 2 的 task id；stage 1 可传 None
    输出:
        g:        [B, N, 1]   -> 融合系数
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        gate_max_val: float = 1.0,
        fixed_min_val: float = 0.5,
        learn_task_floor: bool = False,
        min_floor_limit: float = 0.30,
        delta_scale: float = 0.10,
        init_task_floor_logit: float = -6.0,
        block_idx: int = -1,
    ):
        super().__init__()

        assert gate_max_val > fixed_min_val
        assert 0.0 <= min_floor_limit <= fixed_min_val
        assert fixed_min_val <= 1.0

        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.gate_max_val = gate_max_val
        self.fixed_min_val = fixed_min_val
        self.learn_task_floor = learn_task_floor
        self.min_floor_limit = min_floor_limit
        self.delta_scale = delta_scale

        # token residual branch
        self.gate_linear = nn.Linear(input_dim, 1)

        # task-specific base gate: (T, 1)
        self.task_gate = nn.Parameter(torch.zeros(num_tasks, 1))

        # task-specific floor control: (T, 1)
        # 只有 deep block 建议打开 learn_task_floor=True
        if learn_task_floor:
            self.task_floor_logits = nn.Parameter(
                torch.full((num_tasks, 1), init_task_floor_logit)
            )
        else:
            self.register_parameter("task_floor_logits", None)

        self.block_idx = block_idx
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.zeros_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.zeros_(self.task_gate)

        with torch.no_grad():

            if self.block_idx in [2, 5]:  # 浅层
                self.task_gate[:, 0] = torch.tensor([
                    0.6,  # semseg
                    0.5,  # human_parts
                    0.2,  # normals
                    0.1,  # edge
                    0.55,  # sal
                ], device=self.task_gate.device)

            else:  # 深层 (8, 11)
                self.task_gate[:, 0] = torch.tensor([
                    0.5,  # semseg
                    0.3,  # human_parts
                    -0.1,  # normals
                    -0.3,  # edge（稍微收一点）
                    0.4,  # sal
                ], device=self.task_gate.device)

        if self.task_floor_logits is not None:
            with torch.no_grad():
                if self.block_idx in [8, 11]:
                    self.task_floor_logits[:, 0] = torch.tensor([
                        -6.0,  # semseg
                        -6.0,  # human_parts
                        -1.5,  # normals
                        -0.5,  # edge
                        -6.0,  # sal
                    ], device=self.task_floor_logits.device)
                else:
                    self.task_floor_logits[:, 0].fill_(-6.0)

    def _compute_min_g(self, task_ids: torch.Tensor, B: int, device, dtype):
        """
        返回 [B, 1, 1] 的 min_g
        """
        if (task_ids is None) or (not self.learn_task_floor):
            min_g = torch.full(
                (B, 1, 1),
                fill_value=self.fixed_min_val,
                device=device,
                dtype=dtype,
            )
            return min_g

        # task-specific floor:
        # floor 从 fixed_min_val 最多往下放到 min_floor_limit
        # sigmoid(logit) in (0,1), 因此:
        # min_g = fixed_min_val - (fixed_min_val - min_floor_limit) * sigmoid(...)
        # 初始 logits=-6 -> sigmoid很小 -> min_g ≈ fixed_min_val，很稳
        floor_strength = torch.sigmoid(self.task_floor_logits[task_ids])  # [B,1]
        min_g = self.fixed_min_val - (self.fixed_min_val - self.min_floor_limit) * floor_strength
        min_g = min_g[:, None, :]  # [B,1,1]
        return min_g.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor = None, return_stats: bool = False):
        """
        x: [B, N, C]
        task_ids: [B] or None
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype


        # task-specific base term
        if task_ids is None:
            base = torch.zeros(B, 1, 1, device=device, dtype=dtype)
        else:
            base = self.task_gate[task_ids]   # [B,1]
            base = base[:, None, :]           # [B,1,1]
            base = base.to(device=device, dtype=dtype)

        # small token-wise residual
        delta = self.delta_scale * torch.tanh(self.gate_linear(x))   # [B,N,1]

        gate_pre_activation = base + delta
        min_g = self._compute_min_g(task_ids, B, device, dtype)

        # dynamic constrained sigmoid:
        # g in [min_g, gate_max_val]
        s = torch.sigmoid(gate_pre_activation)
        g = min_g + (self.gate_max_val - min_g) * s

        if return_stats:
            stats = {
                "gate_mean": g.mean().detach(),
                "gate_min": g.min().detach(),
                "gate_max": g.max().detach(),
                "preact_mean": gate_pre_activation.mean().detach(),
                "floor_mean": min_g.mean().detach(),
                "near_floor_ratio": (g <= (min_g + 0.02)).float().mean().detach(),
            }
            return g, stats

        return g, gate_pre_activation

# 在 GatedMoETransformerBlock 中
# self.gate = SmarterGate(dim, patch_resolution)
# global_print("Using no structure distill!")
########################no structure distill#########################

class RSL(nn.Module):
    def __init__(self, in_features, num_experts, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.num_experts = num_experts
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_experts),
        )
        # **Modify initialization method**
        nn.init.xavier_normal_(self.mlp[0].weight, gain=math.sqrt(2))  # Suitable for GELU
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_normal_(self.mlp[2].weight, gain=1.0)  # Also can be set to 1.0

    def forward(self, hidden_states):
        B, seq_len, hidden_dim = hidden_states.shape
        logits = self.mlp(hidden_states).float()
        scores = F.softmax(logits, dim=-1)

        # top3
        top3_weights, top3_indices = torch.topk(scores, k=3, dim=-1)
        top3_weights = F.normalize(top3_weights, p=1, dim=-1)

        aux_loss = None
        if self.training and self.alpha > 0.0:
            Pi = scores.mean(dim=0)  # Mean over tokens
            fi = F.one_hot(top3_indices.view(-1), num_classes=self.num_experts).float().mean(dim=0)

            balance = (Pi * fi).sum()
            entropy = - (scores * scores.clamp(min=1e-9).log()).sum(dim=-1).mean()

            gamma = 0.1  # Control selectivity preference, suggested range 0.01 ~ 0.1
            aux_loss = self.alpha * balance - gamma * entropy

        router_weights = scores.view(B, seq_len, self.num_experts).float()
        return router_weights, aux_loss, top3_indices, top3_weights


class ConditionedMoELayer(nn.Module):
    """条件化的MoE层"""

    def __init__(self, input_dim, num_experts, condition_dim, top_k=1,
                 expert_hidden_ratio=4.0,
                 load_balancing_alpha=0.01, noisy_gating=True, num_experts_per_group=5,
                 router_type="no_structure_router",
                 base_block=None,
                 # --- 新增 Loss 参数 ---
                 ortho_loss_weight=0.01,  # 正交损失权重
                 variance_loss_weight=0.01,  # 方差损失权重
                 seq_aux=False,  # 是否使用序列级辅助损失
                 num_tasks=0,
                 ):
        # dim: 特征维度
        # num_experts: 通用Expert数量
        # condition_dim: 条件嵌入维度
        # top_k: 每个token选择top_k个Expert
        # expert_hidden_ratio: Expert内部FFN的隐层比例
        # load_balancing_alpha: 负载均衡损失的系数

        # 参数验证

        super().__init__()
        self.dim = input_dim
        self.num_experts = num_experts
        self.condition_dim = condition_dim
        self.top_k = top_k
        self.load_balancing_alpha = load_balancing_alpha
        self.num_experts_per_group = num_experts_per_group  # <--- 保存参数

        # --- 新增 Loss 配置 ---
        self.ortho_loss_target = ortho_loss_weight
        self.variance_loss_target = variance_loss_weight
        self.seq_aux = seq_aux

        # 2. 初始化权重为 -1 (表示尚未初始化)
        self.register_buffer('ortho_loss_weight', torch.tensor(-1.0))
        self.register_buffer('variance_loss_weight', torch.tensor(-1.0))

        new_weight = self._compute_dynamic_weight(torch.tensor(0.004540108144283295), self.variance_loss_target)
        self.variance_loss_weight.fill_(new_weight)

        new_weight = self._compute_dynamic_weight(torch.tensor(0.9623544216156006), self.ortho_loss_target)
        self.ortho_loss_weight.fill_(new_weight)

        if router_type == "structure_router":
            self.router_type = "structure_router"
            if num_experts % num_experts_per_group != 0:
                raise ValueError("num_experts must be divisible by num_experts_per_group.")

            self.router = StructureConditionedRouter(input_dim, num_experts, condition_dim, noisy_gating, top_k=self.top_k)
        elif router_type == "no_structure_router":
            self.router_type = "no_structure_router"
            self.router = ConditionedRouter(input_dim, num_experts, condition_dim, noisy_gating)
        elif router_type == "advanced_router":
            self.router_type = "advanced_router"
            self.router = AdvancedConditionedRouter(input_dim, num_experts, condition_dim, noisy_gating)
        elif router_type == "decoupled_router":
            self.router_type = "decoupled_router"
            if num_experts % num_experts_per_group != 0:
                raise ValueError("num_experts must be divisible by num_experts_per_group.")

            self.router = DecoupledGroupRouter(input_dim, num_experts, num_experts_per_group,  condition_dim, noisy_gating, num_tasks=num_tasks)
        self.experts = nn.ModuleList(
            [UniversalExpert(input_dim, expert_hidden_ratio) for _ in range(num_experts)]
        )
        # noise_std = 0.02
        for i, expert in enumerate(self.experts):
            self.init_expert_from_base_mlp(expert, base_block.mlp)
            # expert.load_state_dict(copy.deepcopy(base_block.mlp.state_dict()))
            # # B. 【核心修改】对权重加噪声 (Symmetry Breaking)
            # with torch.no_grad():
            #     # 对 fc1 (输入层) 加噪声是最关键的，因为它直接影响专家对特征的敏感度
            #     expert.fc1.weight.add_(torch.randn_like(expert.fc1.weight) * noise_std)
            #
            #     # fc2 (输出层) 也可以加一点
            #     expert.fc2.weight.add_(torch.randn_like(expert.fc2.weight) * noise_std)

            # 偏置 (bias) 通常不需要加噪声，或者加很小的噪声

        # --- 方案 B 的核心：超级专家/融合层 ---
        # 它在稀疏专家路径之后，对结果进行非线性融合
        # self.super_expert_fusion_layer = UniversalExpert(input_dim, expert_hidden_ratio)

        # # --- 暂时禁用或移除你原来的shared_expert ---
        self.shared_expert = UniversalExpert(input_dim, expert_hidden_ratio)
        self.init_expert_from_base_mlp(self.shared_expert,base_block.mlp)
        # self.shared_expert.load_state_dict(copy.deepcopy(base_block.mlp.state_dict()))



        # self.shared_expert = SpatialSharedExpert(input_dim, expert_hidden_ratio)
        # missing_keys, unexpected_keys = self.shared_expert.load_state_dict(
        #     copy.deepcopy(base_block.mlp.state_dict()), strict=False
        # )
        #
        # # 【关键 Trick：Dirac 初始化】
        # # 让 3x3 卷积的中心权重为 1，周围为 0。
        # # 这样在初始状态下，卷积层就等于一个 Identity (恒等映射)，
        # # 完全不会破坏原有的 ImageNet 预训练知识，模型会平滑地慢慢学会提取边缘。
        # with torch.no_grad():
        #     nn.init.dirac_(self.shared_expert.dwconv.weight)
        #     if self.shared_expert.dwconv.bias is not None:
        #         nn.init.zeros_(self.shared_expert.dwconv.bias)

    def forward(self, x, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None, task_id=None, return_expert_outputs=False):
        """
        x: (B, N, D)
        """
        batch_size, num_tokens, dim = x.shape
        x_flat = x.view(-1, dim)  # (B*N, D)
        current_top_k = self.top_k


        # ============================================
        # === 1. Shared Expert Path (共享专家路径) ===
        # ============================================
        shared_output = self.shared_expert(x)

        # ============================================
        # === 2. Sparse Experts Path (稀疏专家路径) ===
        # ============================================

        # 预定义变量，用于统一不同 Router 分支的输出
        gating_weights = None  # (B*N, top_k)
        top_k_indices = None  # (B*N, top_k)
        router_probs_flat = None  # (B*N, num_experts) -> 用于计算 Aux Loss

        # ---------------------------------------------------------------------
        # 分支 A: Structure Router (新逻辑)
        # ---------------------------------------------------------------------
        if self.router_type == "structure_router":
            # --- 2.1 创建路由掩码 (Phase 1 Only) ---
            routing_mask = None
            # 1. 初始化动态参数
            routing_mask = None
            # 默认使用下游任务的 top_k (如 3 或 4)

            # 2. 如果当前是 VFM 蒸馏流 (无论在 Phase 1 还是 Phase 2)
            if is_vfm_condition:
                if vfm_teacher_id is None:
                    raise ValueError("vfm_teacher_id must be provided in vfm_condition mode.")

                # 蒸馏路径强制使用 top_k = 2
                current_top_k = 2

                # 矢量化创建掩码: (B, num_experts) -> 物理隔离
                group_indices = vfm_teacher_id.unsqueeze(1)
                expert_indices = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
                start_indices = group_indices * self.num_experts_per_group
                is_in_group = (expert_indices >= start_indices) & (
                        expert_indices < start_indices + self.num_experts_per_group)
                routing_mask = torch.where(is_in_group, 0.0, float('-inf'))

            # 3. 调用 Router (传入动态 top_k)
            _top_k_probs, _top_k_indices, _router_probs = self.router(
                x,
                condition_embedding=condition_embedding,
                is_vfm_condition=is_vfm_condition,
                routing_mask=routing_mask,
                override_top_k=current_top_k  # 传入动态 k
            )

            # 4. 获取展平后的权重和索引
            # 注意：这里的 gating_weights 宽度是 current_top_k
            gating_weights = _top_k_probs.view(-1, current_top_k)
            top_k_indices = _top_k_indices.view(-1, current_top_k)
            router_probs_flat = _router_probs.view(-1, self.num_experts)

            # # --- 2.2 调用新 Router ---
            # # 新 Router 内部处理了 FiLM, Mask, Noise, Softmax 和 TopK
            # # 返回: (B, N, k), (B, N, k), (B, N, E)
            # _top_k_probs, _top_k_indices, _router_probs = self.router(
            #     x,
            #     condition_embedding,
            #     is_vfm_condition=is_vfm_condition,
            #     routing_mask=routing_mask,
            # )
            #
            # # 展平以适配后续计算
            # gating_weights = _top_k_probs.view(-1, self.top_k)
            # top_k_indices = _top_k_indices.view(-1, self.top_k)
            # router_probs_flat = _router_probs.view(-1, self.num_experts)

        # ---------------------------------------------------------------------
        # 分支 B: No Structure Router (你原来的逻辑，保持不变)
        # ---------------------------------------------------------------------
        elif self.router_type == "no_structure_router":
            # 这里的 router_logits 应该是 (B, N, E)
            router_logits = self.router(
                x,
                condition_embedding,
                is_vfm_condition=is_vfm_condition,
                routing_mask=None
            )
            router_logits_flat = router_logits.view(-1, self.num_experts)

            # 手动 Top-K
            top_k_logits, top_k_indices = torch.topk(router_logits_flat, self.top_k, dim=-1)
            gating_weights = F.softmax(top_k_logits, dim=-1)  # (B*N, k)

            # 手动 Softmax 用于 Loss
            router_probs_flat = F.softmax(router_logits_flat, dim=-1)

        elif self.router_type == "advanced_router":
            # --- 2. 调用 Advanced Router ---
            # 输入: x, condition, mask
            # 输出: Logits (B, N, E) -- 注意这里返回的是分数，不是概率
            router_logits = self.router(
                x,
                condition_embedding,
                is_vfm_condition=is_vfm_condition,
                routing_mask=None
            )

            # --- 3. 后处理 (Flatten & Top-K) ---
            # 展平以适配后续计算 (B*N, E)
            router_logits_flat = router_logits.view(-1, self.num_experts)

            # 计算完整的概率分布 (用于 Aux Loss 计算)
            router_probs_flat = F.softmax(router_logits_flat, dim=-1)

            # 执行 Top-K 选择
            # 如果使用了 routing_mask (-inf), topk 会自动跳过被 mask 的专家
            top_k_logits, top_k_indices = torch.topk(router_logits_flat, self.top_k, dim=-1)

            # 对选中的 Top-K Logits 进行 Softmax 归一化，作为最终的分发权重
            gating_weights = F.softmax(top_k_logits, dim=-1)  # (B*N, k)

        elif self.router_type == "decoupled_router":
            # 2. Router
            # 注意：这里我们移除了 routing_mask 的逻辑，因为 DecoupledRouter 天然隔离
            # Stage 1 时，task_id 为 None，Router 返回无偏置的组内 Top-1
            if task_id == None:
                override_top_k = 2
            else:
                override_top_k = 1
            gating_weights, top_k_indices, probs_grouped = self.router(
                x, condition_embedding, task_id, override_top_k
            )
            # ==================================================
            # === Stage 1: 强制单组路由 (Enforce Single Group) ===
            # ==================================================
            if is_vfm_condition and vfm_teacher_id is not None:
                # 1. 对齐 vfm_teacher_id 到 Token 级别
                if vfm_teacher_id.shape[0] == batch_size:
                    vfm_teacher_id_flat = vfm_teacher_id.repeat_interleave(num_tokens, dim=0)
                else:
                    vfm_teacher_id_flat = vfm_teacher_id

                # 2. 生成 Group Mask (B*N, Groups) -> (B*N, 3)
                group_mask = F.one_hot(vfm_teacher_id_flat, num_classes=self.router.num_groups).float()

                # 3. 【核心修正】扩展 Mask 以匹配 Top-K > 1
                # gating_weights 形状是 (B*N, G*K)，例如 6 列
                # group_mask 形状是 (B*N, G)，例如 3 列
                current_total_k = gating_weights.shape[1]
                k_per_group = current_total_k // self.router.num_groups  # e.g. 2

                if k_per_group > 1:
                    # [1, 0, 0] -> [1, 1, 0, 0, 0, 0]
                    group_mask = group_mask.repeat_interleave(k_per_group, dim=1)

                # 4. 施加 Mask
                # 非目标组的权重变为 0，目标组的 2 个专家权重保留
                # [关键修复] 重新归一化！否则权重和远小于1，导致信号丢失
                gating_sum = gating_weights.sum(dim=-1, keepdim=True) + 1e-6
                gating_weights = gating_weights / gating_sum
            # gating_weights: (B*N, 3)
            # top_k_indices: (B*N, 3)

            # 3. Dispatch & Compute (复用你原来的逻辑，完全兼容)
            # 此时 self.num_experts (15) 应该对应 gating_weights 的列数吗？
            # 不！gating_weights 只有 3 列 (因为每组选1个)。
            # 但 dispatch 需要映射回 15 个专家的空间。
            #
            # F.one_hot(indices, 15) -> (B*N, 3, 15)
            # gating_weights.unsqueeze(-1) -> (B*N, 3, 1)
            # dispatch_weights = F.one_hot(top_k_indices, self.num_experts).float() * gating_weights.unsqueeze(-1)
            # # sum dim=1 -> (B*N, 15)
            # combine_weights = dispatch_weights.sum(dim=1)
            #
            # # ... (标准的 Einsum 计算逻辑，保持不变) ...
            # dispatched_input = torch.einsum('be,bd->bed', combine_weights, x_flat)
            # dispatched_input = dispatched_input.transpose(0, 1)
            # expert_outputs = torch.stack([self.experts[i](dispatched_input[i]) for i in range(self.num_experts)])
            # expert_outputs = expert_outputs.transpose(0, 1)
            # sparse_output_flat = torch.einsum('be,bed->bd', combine_weights, expert_outputs)
            # sparse_output = sparse_output_flat.view(batch_size, num_tokens, dim)
            #
            # final_output = shared_output + sparse_output




        # ============================================
        # === 3. 分发和计算 (Dispatch & Compute) ===
        # ============================================
        # 以下代码两个分支共用
        # if self.training: # and self.expert_dropout_rate > 0.0:
        #     # expert_dropout_rate 建议设为 0.1 到 0.2
        #
        #     # 生成一个随机掩码，以一定概率丢弃专家的激活
        #     # mask shape: (Num_Experts,) -> 广播到 (B*N, Num_Experts)
        #     # 我们希望是对每个 Token 独立 dropout，还是对每个专家整体 dropout？
        #     # 推荐：对每个 Expert 独立 Dropout (随机关掉某个专家)
        #     self.expert_dropout_rate =
        #     mask = torch.rand(self.num_experts, device=x.device) > self.expert_dropout_rate
        #     # mask: [1, 1, 0, 1, ...] (0 表示该专家在本轮 forward 中彻底休息)
        #
        #     # 应用掩码
        #     dispatch_weights = dispatch_weights * mask.view(1, -1)
        #
        #     # 重新归一化 (可选，保持数值稳定)
        #     # dispatch_weights = dispatch_weights / (dispatch_weights.sum(dim=-1, keepdim=True) + 1e-6)
        # 1. 创建 One-hot 分发权重: (B*N, top_k, E) * (B*N, top_k, 1)

        # =========================================================
        # === 动态权重更新 1: Variance Loss Weight ===
        # =========================================================
        # # 如果是第一次运行 (权重为 -1)，且在训练模式
        # if self.training and self.variance_loss_weight == -1 and self.variance_loss_target > 0:
        #     with torch.no_grad():
        #         # 计算当前的方差
        #         current_var = torch.var(router_probs_flat, dim=-1).mean()
        #         # 动态计算权重并保存
        #         # print("current_var is" + str(current_var.item()))
        #         new_weight = self._compute_dynamic_weight(current_var, self.variance_loss_target)
        #         self.variance_loss_weight.fill_(new_weight)
        #         # print(f"Initialized Variance Loss Weight: {self.variance_loss_weight.item()}")

        # ============================================
        # === 3. Sparse Dispatch & Execution (稀疏执行) ===
        # ============================================
        # 核心优化：避免创建 (B*N, E) 的密集 dispatch_weights，改用 combine_weights + Loop

        # 1. 构建全局组合权重 (B*N, num_experts)
        # 这是一个稀疏矩阵的概念，但在 PyTorch 中我们需要一个 Tensor 来存储权重以计算 Load
        combine_weights = torch.zeros(
            batch_size * num_tokens, self.num_experts,
            device=x.device, dtype=x.dtype
        )

        # 将 gating_weights 填入对应专家位置
        # src: (B*N, K), index: (B*N, K) -> dest: (B*N, E)
        combine_weights.scatter_add_(1, top_k_indices, gating_weights)

        # 2. 稀疏循环计算
        sparse_output_flat = torch.zeros_like(x_flat)

        # 如果需要返回 expert_outputs (调试用)，必须非常小心显存
        # 在稀疏模式下，我们通常无法返回完整的 (B, N, E, D) 张量
        # 这里仅作占位，实际很难在不爆显存的情况下返回 dense output
        collected_expert_outputs = None
        if self.ortho_loss_weight > 0 and current_top_k > 1:
            collected_expert_outputs = torch.zeros(
                batch_size * num_tokens, current_top_k, dim,
                device=x.device, dtype=torch.float16
            )

        for i in range(self.num_experts):
            # 获取当前专家的权重列: (B*N,)
            expert_weight = combine_weights[:, i]

            # 找出需要该专家的 Token
            active_mask = expert_weight > 1e-6

            # Extract Input
            active_inputs = x_flat[active_mask]

            # === 修复 2: 模拟旧版 "Double Weighting" ===
            # 旧版代码中: dispatched_input = einsum(weights, x)
            # 这意味着输入 Expert 前，x 已经被权重缩放了。
            # 为了对齐旧版数学逻辑 (从而恢复 70 分的性能)，我们需要在这里乘一次权重。
            # 如果你想用标准 MoE (Input 不缩放)，请注释掉下面这行，但那可能需要重新调参。
            weights_active = expert_weight[active_mask].unsqueeze(-1)
            active_inputs = active_inputs * weights_active
            # =========================================
            # Forward (计算当前专家的原始输出)
            active_outputs = self.experts[i](active_inputs)

            # 找出全局 Token 索引
            indices = torch.nonzero(active_mask).squeeze(-1)

            # --- 【关键步骤：保存用于 Loss 的输出】 ---
            if collected_expert_outputs is not None:
                # 我们需要知道：对于这批 Token，当前专家 i 是它们的第几选择 (0..K-1)?
                # top_k_indices shape: (B*N, K)
                # 取出这批 Token 的 top_k 列表: (M, K)
                current_tokens_topk = top_k_indices[indices]

                # 找到 i 在其中的位置 (mask)
                # is_current_expert shape: (M, K)，每行只有一个 True
                is_current_expert = (current_tokens_topk == i)

                # 获取列索引 (0..K-1)
                # (M, )
                k_indices = is_current_expert.nonzero(as_tuple=True)[1]

                # 填入 collected_expert_outputs
                # x 轴: 全局 token 索引 (indices)
                # y 轴: 第几顺位 (k_indices)
                # z 轴: 向量内容
                collected_expert_outputs[indices, k_indices] = active_outputs.to(collected_expert_outputs.dtype)
            # ----------------------------------------

            # Weighting & Scatter Add (正常的 MoE 输出计算)
            weighted_outputs = active_outputs * expert_weight[active_mask].unsqueeze(-1)
            sparse_output_flat.index_add_(0, indices, weighted_outputs)

        # 3. 恢复形状
        sparse_output = sparse_output_flat.view(batch_size, num_tokens, dim)
        # =====================================
        # === 4. 最终合并 ===
        # =====================================

        if return_expert_outputs:
        # --- 2.3 批量化专家计算 ---
            # 将输入与分发权重相乘，得到每个专家的输入
            # einsum('be,bd->bed', combine_weights, x_flat)
            # b: B*N (token), e: expert, d: dim
            dispatched_input = torch.einsum('be,bd->bed', combine_weights, x_flat)

            # 将所有专家的计算看作一个大的批量矩阵乘法
            # 这是一个常见的优化技巧，避免在Python中循环
            # dispatched_input (B*N, E, D) -> (E, B*N, D)
            dispatched_input = dispatched_input.transpose(0, 1)

            # expert_outputs (E, B*N, D)
            expert_outputs = torch.stack([self.experts[i](dispatched_input[i]) for i in range(self.num_experts)])

            # 转置回来 (E, B*N, D) -> (B*N, E, D)
            expert_outputs = expert_outputs.transpose(0, 1)
        #

        final_output = shared_output + sparse_output
        # final_output = sparse_output
        dispatch_weights = combine_weights
        # # =====================================
        # # === 5. Decoupled Router Loss 计算 (针对新架构调整) ===
        # # =====================================
        # # =====================================
        # # === 组内 Loss 计算 (Intra-Group) ===
        # # =====================================
        #
        # load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        #
        # if self.training:
        #     # 1. 组内 Lo & Lv (Load Balancing & Variance)
        #     if self.load_balancing_alpha > 0 or self.variance_loss_weight > 0:
        #
        #         for i in range(self.router.num_groups):
        #             # --- 准备数据 ---
        #             # group_probs: (B*N, K) -> 该组内的概率分布
        #             group_probs = probs_grouped[:, i, :]
        #
        #             # --- A. 组内负载均衡 (Aux Loss) ---
        #             if self.load_balancing_alpha > 0:
        #                 start = i * self.num_experts_per_group
        #                 end = start + self.num_experts_per_group
        #
        #                 # group_weights: (B*N, K) -> 实际分发权重
        #                 group_weights = combine_weights[:, start:end]
        #
        #                 # load: (K,)
        #                 group_load = (group_weights > 0).float().mean(dim=0)
        #                 group_prob_mean = group_probs.mean(dim=0)
        #
        #                 # 标准 Aux Loss 公式
        #                 loss_aux_i = torch.sum(group_load * group_prob_mean) * self.num_experts_per_group
        #                 load_balancing_loss += self.load_balancing_alpha * loss_aux_i
        #
        #             # # --- B. 组内方差损失 (Variance Loss) ---
        #             # if self.variance_loss_weight > 0:
        #             #     # 最大化方差 -> 最小化负方差
        #             #     variance = torch.var(group_probs, dim=-1).mean()
        #             #     load_balancing_loss -= self.variance_loss_weight * variance
        #
        #     # 2. 组内参数正交损失 (Weight Ortho Loss)
        #     # 注意：这部分不需要输入数据 x，只看参数
        #     if self.ortho_loss_weight > 0:
        #         for i in range(self.router.num_groups):
        #             start = i * self.num_experts_per_group
        #             end = start + self.num_experts_per_group
        #
        #             # 提取权重: (K, P)
        #             # 假设 Expert 结构是 MLP(fc1, act, fc2)，我们取 fc1 (输入方向) 或 fc2 (输出方向)
        #             # 推荐取 fc1 (Down-Projection)，因为它决定了专家"关注什么特征"
        #             # fc1.weight shape: (Hidden, Dim)
        #             w_list = [e.fc1.weight.view(1, -1) for e in self.experts[start:end]]
        #             w_matrix = torch.cat(w_list, dim=0)
        #
        #             # 归一化 & Gram Matrix
        #             w_norm = F.normalize(w_matrix, p=2, dim=1)
        #             gram = torch.mm(w_norm, w_norm.t())
        #
        #             # 惩罚非对角线元素
        #             eye = torch.eye(self.num_experts_per_group, device=x.device)
        #             loss_ortho_i = torch.mean((gram - eye) ** 2)
        #
        #             load_balancing_loss += self.ortho_loss_weight * loss_ortho_i

        # =====================================
        # === 5. Load Balancing Loss ===
        # =====================================
        ##==========================================================
        # ##=== 改进版 Loss 计算 (RSL Logic: Balance + Entropy) ===
        # # =====================   RSL loss  =====================================
        # aux_loss = torch.tensor(0.0, device=x.device)
        # load_balancing_loss = torch.tensor(0.0, device=x.device)
        # if self.training:
        #     # -----------------------------------------------------
        #     # 1. 准备数据
        #     # -----------------------------------------------------
        #     # router_probs_flat: (B*N, Num_Experts) -> 对应参考代码的 scores
        #     scores = router_probs_flat
        #
        #     # top_k_indices_flat: (B*N, TopK) -> 对应参考代码的 top3_indices
        #     # 将其展平用于计算频率 fi
        #     all_indices = top_k_indices.view(-1)
        #
        #     # -----------------------------------------------------
        #     # 2. 计算 Balance Loss (负载均衡)
        #     # -----------------------------------------------------
        #     if self.load_balancing_alpha > 0:
        #         # Pi: 每个专家在 batch 里的平均预测概率
        #         # shape: (Num_Experts,)
        #         Pi = scores.mean(dim=0)
        #
        #         # fi: 每个专家实际被选中的频率
        #         # F.one_hot 生成 (B*N*K, Num_Experts)，然后求平均
        #         # 这与你提供的 RSL 代码逻辑完全一致
        #         fi = F.one_hot(all_indices, num_classes=self.num_experts).float().mean(dim=0)
        #
        #         # 计算点积
        #         # 标准实现通常会乘以 self.num_experts 以保持数值量级
        #         balance = torch.sum(Pi * fi) * self.num_experts
        #
        #         aux_loss += self.load_balancing_alpha * balance
        #
        #     # -----------------------------------------------------
        #     # 3. 计算 Entropy Loss (最大熵正则化 - 防崩塌核心)
        #     # -----------------------------------------------------
        #     # gamma: 建议设为 0.01 到 0.1。
        #     # 如果 Layer 8 崩塌严重，建议从 0.1 开始尝试。
        #     entropy_gamma = 0.01
        #
        #     if entropy_gamma > 0:
        #         # 计算 Shannon Entropy: H(p) = - sum(p * log(p))
        #         # clamp 防止 log(0) 导致 NaN
        #         entropy = - (scores * scores.clamp(min=1e-9).log()).sum(dim=-1).mean()
        #
        #         # 我们希望 Entropy 越大越好 (分布越均匀越好)
        #         # 所以 Loss 应该是减去 Entropy (或者加上负熵)
        #         aux_loss -= entropy_gamma * entropy
        #     load_balancing_loss = aux_loss

        # # ==========================================================
        # #
        # ======================== Normal loss ==================================

        load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training and self.load_balancing_alpha > 0:
            # 两个分支都准备好了 router_probs_flat

            # expert_load: (E,)
            expert_load = (dispatch_weights > 0).float().mean(dim=0)

            # router_probs_mean: (E,)
            router_probs_mean = router_probs_flat.mean(dim=0)

            load_balancing_loss = self.load_balancing_alpha * self.num_experts * torch.sum( expert_load * router_probs_mean )
        #


            # # B. 【核心】软引导 Grouping Loss (仅在 Phase 1 启用)
            # if is_vfm_condition:
            #     # 目标：构建一个 Mask，指示每个样本"应该"去哪个专家组
            #     # vfm_teacher_id: (B,) -> [0, 1, 0, ...]
            #
            #     # 1. 构建 Batch 级别的 Target Mask (B, Num_Experts)
            #     group_indices = vfm_teacher_id.unsqueeze(1)  # (B, 1)
            #     expert_indices = torch.arange(self.num_experts, device=x.device).unsqueeze(0)  # (1, 15)
            #
            #     start_indices = group_indices * self.num_experts_per_group
            #     # valid_mask: 如果专家属于当前 Teacher 的组，则为 1 (True)，否则为 0 (False)
            #     target_group_mask = (expert_indices >= start_indices) & (
            #             expert_indices < start_indices + self.num_experts_per_group)
            #     target_group_mask = target_group_mask.float()  # (B, 15)
            #
            #     # 2. 将 Mask 扩展到 Token 级别 (B, 15) -> (B*N, 15)
            #     # router_probs_flat 是 (B*N, 15)
            #     target_group_mask_flat = target_group_mask.repeat_interleave(num_tokens, dim=0)
            #
            #     # 3. 计算"越界"概率 (Penalty for routing outside the group)
            #     # 我们希望 router_probs 集中在 target_group_mask 为 1 的地方
            #     # 所以我们惩罚 mask 为 0 的地方的概率之和
            #
            #     # invalid_mask = 1 - target_group_mask
            #     # loss = sum(probs * invalid_mask)
            #     prob_outside_group = (router_probs_flat * (1.0 - target_group_mask_flat)).sum(dim=-1)
            #
            #     grouping_loss = prob_outside_group.mean()
            #
            #     load_balancing_loss += 0.1 * grouping_loss



        #
        # # ==============================================================
        # # === 5. lv & lo Loss Calculation (集成 DeepSeek Loss 策略) ===
        # # ==============================================================
        #
        # load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # if self.training:
        #     # --- A. 负载均衡损失 (Aux Loss) ---
        #     if self.load_balancing_alpha > 0:
        #         if self.seq_aux:
        #             # 序列级 Aux Loss (更精细)
        #             scores_for_seq_aux = router_probs_flat.view(batch_size, num_tokens, -1)
        #             topk_idx_flat = top_k_indices.view(batch_size, -1)  # flatten topk to [B, N*K]
        #             ce = torch.zeros(batch_size, self.num_experts, device=x.device)
        #             # 统计每个 Expert 在该序列中被选中的次数
        #             ce.scatter_add_(1, topk_idx_flat, torch.ones(batch_size, num_tokens * self.top_k, device=x.device))
        #             ce.div_(num_tokens * self.top_k / self.num_experts)
        #             aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
        #             load_balancing_loss += self.load_balancing_alpha * aux_loss
        #         else:
        #             # 标准 Batch 级 Aux Loss
        #             expert_load = (dispatch_weights > 0).float().mean(dim=0)  # (E,)
        #             router_probs_mean = router_probs_flat.mean(dim=0)  # (E,)
        #             aux_loss = self.num_experts * torch.sum(expert_load * router_probs_mean)
        #             load_balancing_loss += self.load_balancing_alpha * aux_loss
        #
        #     # --- B. 方差损失 (Variance Loss) ---
        #     # 鼓励 Router 对某些专家打高分，对其他打低分 (One-hot 倾向)，而不是平均分配
        #     if self.variance_loss_target > 0:
        #         # 计算每个 token 在所有专家上的概率方差
        #         variance_loss = torch.var(router_probs_flat, dim=-1).mean()
        #         # 我们希望方差越大越好(分布越尖锐)，但在 Loss 里通常写成目标值除以当前方差，或者负方差
        #         # 参考 DeepSeek 代码逻辑：weight = target / var，所以 loss = var * (target/var) = const?
        #         # 不，通常做法是：Loss = - variance，或者 Loss = exp(-variance)
        #         # 这里根据你提供的代码逻辑，它似乎是用动态权重来平衡。
        #         # 简化起见，我们直接最大化方差 (即最小化负方差)
        #         # 或者使用 DeepSeek 的 trick：不直接优化方差，而是优化一个与方差相关的正则项
        #         # 这里我们采用一个简单有效的实现：鼓励熵最小化 (Entropy Loss) 实际上就等同于方差最大化
        #         # 但为了贴合 DeepSeek 命名，我们使用负方差:
        #         load_balancing_loss -= self.variance_loss_weight * variance_loss
        #
        #     # --- C. 正交损失 (Orthogonality Loss) ---
        #     # 关键：修复数据对齐问题。必须提取 Top-K 对应的专家输出。
        #     if self.ortho_loss_target > 0 : # and self.top_k > 1:
        #         # expert_outputs_all: (B*N, Num_Experts, Dim)
        #         # top_k_indices: (B*N, TopK)
        #
        #         # 我们需要从 dim=1 (Experts) 中 gather 出 top_k_indices 指定的向量
        #         # 扩展 indices 以匹配 gather 维度: (B*N, TopK, Dim)
        #         expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        #
        #         # Gather 操作：得到 [B*N, TopK, Dim]
        #         # 这一步确保了 selected_expert_outputs 的第 k 行就是该 Token 选中的第 k 个专家的输出
        #         selected_expert_outputs = torch.gather(expert_outputs, 1, expanded_indices)
        #
        #         # if self.ortho_loss_weight == -1:
        #         #     with torch.no_grad():
        #         #         # 计算原始正交损失
        #         #         raw_ortho_loss = self.compute_ortho_loss(selected_expert_outputs)
        #         #         # print("raw_ortho_loss is" + str(raw_ortho_loss.item()))
        #         #         # 动态计算权重: weight = 10^round(log10(target / raw))
        #         #         new_ortho_weight = self._compute_dynamic_weight(raw_ortho_loss, self.ortho_loss_target)
        #         #         # 更新 buffer (会自动同步到 device)
        #         #         self.ortho_loss_weight.fill_(new_ortho_weight)
        #         #         # print(f"Initialized Ortho Weight: {self.ortho_loss_weight.item()}")
        #
        #         # 计算正交损失
        #         ortho_loss = self.compute_ortho_loss(selected_expert_outputs)
        #         load_balancing_loss += self.ortho_loss_weight * ortho_loss



        # # =====================================
        # # === grouping Loss + lv & lo 计算模块 ===
        # # =====================================
        #
        # load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        #
        # if self.training:
        #     # -------------------------------------------------------------------------
        #     # 1. Grouping Loss (Phase 1 核心: 宏观隔离)
        #     # -------------------------------------------------------------------------
        #     if is_vfm_condition:
        #         # ... (保持你原有的 Grouping Loss 代码不变) ...
        #         group_indices = vfm_teacher_id.unsqueeze(1)
        #         expert_indices = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
        #         start_indices = group_indices * self.num_experts_per_group
        #         target_group_mask = (expert_indices >= start_indices) & (
        #                 expert_indices < start_indices + self.num_experts_per_group)
        #         target_group_mask = target_group_mask.float()
        #         target_group_mask_flat = target_group_mask.repeat_interleave(num_tokens, dim=0)
        #         prob_outside_group = (router_probs_flat * (1.0 - target_group_mask_flat)).sum(dim=-1)
        #         grouping_loss = prob_outside_group.mean()
        #         load_balancing_loss += 0.1 * grouping_loss
        #
        #     # -------------------------------------------------------------------------
        #     # 2. 组内 Lo & Lv (关键修复：维度匹配)
        #     # -------------------------------------------------------------------------
        #
        #     # [步骤 A] 先计算全局的 Expert Load (15,)
        #     # dispatch_weights 可能是 (B, N, E) 或 (B*N, E)。为了安全，先 view 成 (-1, 15)
        #     dispatch_weights_flat = dispatch_weights.view(-1, self.num_experts)
        #
        #     # 计算每个专家在 batch 内被选中的频率 (15,)
        #     # (dispatch_weights_flat > 0) 产生 bool 掩码，float() 转为 0/1
        #     expert_load_global = (dispatch_weights_flat > 0).float().mean(dim=0)
        #
        #     # 计算全局概率均值 (15,) (router_probs_flat 已经是 B*N, 15)
        #     prob_mean_global = router_probs_flat.mean(dim=0)
        #
        #     # [步骤 B] 循环计算组内 Loss
        #     num_groups = self.num_experts // self.num_experts_per_group
        #
        #     for g_id in range(num_groups):
        #         start_idx = g_id * self.num_experts_per_group
        #         end_idx = start_idx + self.num_experts_per_group
        #
        #         # --- A. 组内负载均衡 (Intra-Group Aux Loss) ---
        #         if self.load_balancing_alpha > 0:
        #             # [关键修复]：直接从全局向量中切片，确保维度是 (5,)
        #             prob_mean_group = prob_mean_global[start_idx:end_idx]  # Shape: (5,)
        #             load_mean_group = expert_load_global[start_idx:end_idx]  # Shape: (5,)
        #
        #             # 现在 prob (5,) 和 load (5,) 可以相乘了
        #             intra_aux_loss = self.num_experts_per_group * torch.sum(prob_mean_group * load_mean_group)
        #
        #             # 累加 Loss (除以组数做归一化)
        #             load_balancing_loss += (self.load_balancing_alpha * intra_aux_loss) / num_groups
        #
        #         # --- B. 组内方差损失 (Intra-Group Variance Loss) ---
        #         if self.variance_loss_weight > 0:
        #             # 切片取出当前组的概率矩阵 (B*N, 5)
        #             group_probs_flat = router_probs_flat[:, start_idx:end_idx]
        #
        #             # 计算该组内的方差
        #             variance = torch.var(group_probs_flat, dim=-1).mean()
        #             load_balancing_loss -= (self.variance_loss_weight * variance) / num_groups
        #
        #     # --- C. 正交损失 (Orthogonality Loss) ---
        #     # 正交损失通常计算在 Top-K 选中的专家之间。
        #     # 为了避免冲突，我们最好只计算"组内正交"，即：
        #     # 只有当选中的两个专家属于同一个组时，才惩罚它们的相似度。
        #
        #     if self.ortho_loss_target > 0: # and self.top_k > 1:
        #         # expert_outputs_all: (B*N, 15, D) (需要保证你得保存了这个中间变量)
        #         # top_k_indices: (B*N, K)
        #
        #         # 1. 提取 Top-K 专家的输出向量
        #         # expanded_indices: (B*N, K, D)
        #         expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        #         # selected_vectors: (B*N, K, D)
        #         selected_vectors = torch.gather(expert_outputs, 1, expanded_indices)
        #
        #         # if self.ortho_loss_weight == -1:
        #         #     with torch.no_grad():
        #         #         # 计算原始正交损失
        #         #         raw_ortho_loss = self.compute_ortho_loss(selected_expert_outputs)
        #         #         print("raw_ortho_loss is" + str(raw_ortho_loss.item()))
        #         #         # 动态计算权重: weight = 10^round(log10(target / raw))
        #         #         new_ortho_weight = self._compute_dynamic_weight(raw_ortho_loss, self.ortho_loss_target)
        #         #         # 更新 buffer (会自动同步到 device)
        #         #         self.ortho_loss_weight.fill_(new_ortho_weight)
        #         #         # print(f"Initialized Ortho Weight: {self.ortho_loss_weight.item()}")
        #         # 2. 归一化向量
        #         selected_vectors = F.normalize(selected_vectors, p=2, dim=-1)
        #
        #         # 3. 计算两两余弦相似度矩阵 (B*N, K, K)
        #         cosine_matrix = torch.bmm(selected_vectors, selected_vectors.transpose(1, 2))
        #
        #         # 4. 构建组内 Mask (Intra-Group Mask)
        #         # 我们只惩罚同一组内的专家重叠。如果 Top-K 跨组了，不惩罚。
        #         # top_k_indices: (B*N, K) -> // experts_per_group -> 得到 Group ID
        #         group_ids = top_k_indices // self.num_experts_per_group  # (B*N, K)
        #
        #         # creating mask: (B*N, K, K)
        #         # mask[b, i, j] = 1 if group_id[b, i] == group_id[b, j] else 0
        #         g_i = group_ids.unsqueeze(2)  # (B*N, K, 1)
        #         g_j = group_ids.unsqueeze(1)  # (B*N, 1, K)
        #         intra_group_mask = (g_i == g_j).float()
        #
        #         # 排除对角线 (自己和自己正交没意义)
        #         eye_mask = torch.eye(self.top_k, device=x.device).unsqueeze(0)
        #         mask = intra_group_mask * (1 - eye_mask)
        #
        #         # 5. 计算带 Mask 的正交 Loss
        #         # 只惩罚属于同一组且非对角线的相似度
        #         ortho_loss = (cosine_matrix * mask).sum() / (mask.sum() + 1e-6)
        #
        #         load_balancing_loss += self.ortho_loss_weight * ortho_loss

        # # -------------------------------------------------------------------------
        # # 5. structure router + 组内 Lo & Lv (Intra-Group Load Balancing & Variance)
        # # -------------------------------------------------------------------------
        #
        # load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        #
        # if self.training:
        #     # -------------------------------------------------------------------------
        #     # 1. 组内 Lo & Lv (Intra-Group Load Balancing & Variance)
        #     # -------------------------------------------------------------------------
        #     # 注意：我删除了原本的 "A. 负载均衡损失"，因为那是全局的，会破坏 Grouping。
        #
        #     # [准备工作] 先计算全局的统计量，避免循环内维度错误
        #     # dispatch_weights: (B, N, E) -> (-1, 15)
        #     dispatch_weights_flat = dispatch_weights.view(-1, self.num_experts)
        #
        #     # expert_load_global: 每个专家被选中的频率 (15,)
        #     expert_load_global = (dispatch_weights_flat > 0).float().mean(dim=0)
        #
        #     # prob_mean_global: 每个专家的平均路由概率 (15,)
        #     prob_mean_global = router_probs_flat.mean(dim=0)
        #
        #     # [循环计算] 针对每个 VFM 组单独计算 Loss
        #     num_groups = self.num_experts // self.num_experts_per_group
        #
        #     for g_id in range(num_groups):
        #         start_idx = g_id * self.num_experts_per_group
        #         end_idx = start_idx + self.num_experts_per_group
        #
        #         # --- A. 组内负载均衡 (Intra-Group Aux Loss) ---
        #         if self.load_balancing_alpha > 0:
        #             # 切片：只取当前组的 5 个专家
        #             prob_mean_group = prob_mean_global[start_idx:end_idx]  # (5,)
        #             load_mean_group = expert_load_global[start_idx:end_idx]  # (5,)
        #
        #             # 计算点积：只要求这 5 个专家之间平衡
        #             intra_aux_loss = self.num_experts_per_group * torch.sum(prob_mean_group * load_mean_group)
        #
        #             # 累加 Loss
        #             load_balancing_loss += (self.load_balancing_alpha * intra_aux_loss) / num_groups
        #
        #         # # --- B. 组内方差损失 (Intra-Group Variance Loss) ---
        #         # # 鼓励组内分布尖锐化（One-hot），或者保持多样性
        #         # if self.variance_loss_weight > 0:
        #         #     # 切片取出当前组的概率矩阵 (B*N, 5)
        #         #     group_probs_flat = router_probs_flat[:, start_idx:end_idx]
        #         #
        #         #     # 计算方差并最大化它 (即最小化负方差)
        #         #     variance = torch.var(group_probs_flat, dim=-1).mean()
        #         #     load_balancing_loss -= (self.variance_loss_weight * variance) / num_groups
        #
        #         # -------------------------------------------------------------------------
        #         # 2. 组内正交损失 (Intra-Group Orthogonality Loss)
        #         # -------------------------------------------------------------------------
        #         # 你的 Mask 逻辑非常棒，这里只要保证 expert_outputs 传进来是对的即可
        #         if self.ortho_loss_weight > 0:
        #             weight_ortho_loss = self._compute_weight_ortho_loss()
        #             load_balancing_loss += self.ortho_loss_weight * weight_ortho_loss
        #         # if self.ortho_loss_weight > 0 and current_top_k > 1:
        #             # # 假设 expert_outputs 是所有专家的输出 (B*N, 15, D)
        #             # # 如果显存不够，这里可以用 top_k_indices 重新 gather 一次
        #             #
        #             # # 1. 提取 Top-K 向量
        #             # # expanded_indices: (B*N, K, D)
        #             # expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        #             # # selected_vectors: (B*N, K, D)
        #             # # 注意：这里假设 expert_outputs 是 (B, N, 15, D) 展平后的 (B*N, 15, D)
        #             # # 如果传入的是 sparse_output (B*N, D) 是不行的，必须是原始专家输出堆叠
        #             # # selected_vectors = torch.gather(expert_outputs, 1, expanded_indices)
        #             # selected_vectors = collected_expert_outputs.float()
        #             # # 2. 归一化
        #             # selected_vectors = F.normalize(selected_vectors, p=2, dim=-1)
        #             #
        #             # # 3. 计算相似度矩阵 (B*N, K, K)
        #             # cosine_matrix = torch.bmm(selected_vectors, selected_vectors.transpose(1, 2))
        #             #
        #             # # 4. 构建组内 Mask (核心逻辑)
        #             # # group_ids: (B*N, K) 指示每个选中的专家属于哪个组 (0, 1, 2)
        #             # group_ids = top_k_indices // self.num_experts_per_group
        #             #
        #             # g_i = group_ids.unsqueeze(2)  # (B*N, K, 1)
        #             # g_j = group_ids.unsqueeze(1)  # (B*N, 1, K)
        #             #
        #             # # intra_group_mask: 只有当两个专家属于同一组时为 1
        #             # intra_group_mask = (g_i == g_j).float()
        #             #
        #             # # eye_mask: 排除自己和自己
        #             # eye_mask = torch.eye(current_top_k, device=x.device).unsqueeze(0)
        #             # mask = intra_group_mask * (1 - eye_mask)
        #             #
        #             # # 5. 计算 Loss
        #             # # 只有当 mask.sum() > 0 时才计算，防止除零
        #             # valid_pairs = mask.sum()
        #             # if valid_pairs > 0:
        #             #     ortho_loss = (cosine_matrix * mask).sum() / valid_pairs
        #             #     load_balancing_loss += self.ortho_loss_weight * ortho_loss

        # --- 返回逻辑 ---
        if return_expert_outputs:
            return final_output, load_balancing_loss, top_k_indices, expert_outputs

        if not self.training:
            return final_output, load_balancing_loss, top_k_indices, None

        return final_output, load_balancing_loss, None, None


    # def forward(self, x, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None, return_expert_outputs=False):
    #     """
    #             Args:
    #                 x (torch.Tensor): 输入特征 (B, N, D)
    #                 condition_embedding (torch.Tensor): 条件嵌入 (B, cond_D)
    #                 is_vfm_condition (bool): 是否为第一阶段VFM蒸馏
    #                 vfm_teacher_id (torch.Tensor, optional):
    #                     形状为 (B,) 的VFM教师ID张量。仅在 is_vfm_condition=True 时需要。
    #     """
    #     # x: 输入特征, 形状 (B, N, D)
    #     # condition_embedding: 条件嵌入, 形状 (B, cond_D)
    #     batch_size, num_tokens, dim = x.shape
    #
    #     # ============================================
    #     # === 1. Shared Expert Path (共享专家路径) ===
    #     # ============================================
    #     shared_output = self.shared_expert(x)  # -> (B, N, D)
    #
    #     # ============================================
    #     # === 2. Sparse Experts Path (稀疏专家路径) ===
    #     # ============================================
    #     x_flat = x.view(-1, dim)  # 展平为 (B*N, D)
    #
    #     # --- 2.1 创建路由掩码 (Routing Mask Creation) ---
    #     routing_mask = None
    #     if is_vfm_condition:
    #         if vfm_teacher_id is None:
    #             raise ValueError("vfm_teacher_id must be provided in structured_distill mode.")
    #
    #         # --- 使用矢量化操作高效创建掩码 ---
    #         # (B,) -> (B, 1)
    #         group_indices = vfm_teacher_id.unsqueeze(1)
    #
    #         # (num_experts,) -> (1, num_experts)
    #         expert_indices = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
    #
    #         # 使用广播计算每个专家是否属于每个样本对应的组
    #         # start_indices.shape: (B, 1)
    #         start_indices = group_indices * self.num_experts_per_group
    #         # is_in_group.shape: (B, num_experts)
    #         is_in_group = (expert_indices >= start_indices) & (
    #                     expert_indices < start_indices + self.num_experts_per_group)
    #
    #         # 将布尔掩码转换为浮点掩码 (True -> 0.0, False -> -inf)
    #         routing_mask = torch.where(is_in_group, 0.0, float('-inf'))
    #
    #     # --- 2.2 获取路由门控 (Gating Logic) ---
    #     # 将创建的掩码传递给路由器
    #     router_logits = self.router(
    #         x,
    #         condition_embedding,
    #         is_vfm_condition=is_vfm_condition,
    #         routing_mask=routing_mask
    #     )
    #     router_logits_flat = router_logits.view(-1, self.num_experts)
    #
    #     # 选出 top_k 个专家的 logits 和索引
    #     # 由于被mask掉的logits是-inf, topk会自动忽略它们
    #     top_k_logits, top_k_indices = torch.topk(router_logits_flat, self.top_k, dim=-1)
    #
    #     # 只对 top_k 个 logits 应用 softmax，得到最终的门控权重
    #     gating_weights = F.softmax(top_k_logits, dim=-1)
    #
    #     # --- 2.3 高效地分发和组合 (Dispatch and Combine) ---
    #
    #     # 创建一个稀疏的分发矩阵，值为门控权重
    #     # (B*N, E) 的矩阵，每一行最多有 top_k 个非零值
    #     # F.one_hot -> (B*N, top_k, E)
    #     # gating_weights.unsqueeze(-1) -> (B*N, top_k, 1)
    #     # 广播相乘后，在 top_k 维度上求和
    #     dispatch_weights = F.one_hot(top_k_indices, self.num_experts).float() * gating_weights.unsqueeze(-1)
    #     combine_weights = dispatch_weights.sum(dim=1)  # -> (B*N, E)
    #
    #     # --- 2.3 批量化专家计算 ---
    #     # 将输入与分发权重相乘，得到每个专家的输入
    #     # einsum('be,bd->bed', combine_weights, x_flat)
    #     # b: B*N (token), e: expert, d: dim
    #     dispatched_input = torch.einsum('be,bd->bed', combine_weights, x_flat)
    #
    #     # 将所有专家的计算看作一个大的批量矩阵乘法
    #     # 这是一个常见的优化技巧，避免在Python中循环
    #     # dispatched_input (B*N, E, D) -> (E, B*N, D)
    #     dispatched_input = dispatched_input.transpose(0, 1)
    #
    #     # expert_outputs (E, B*N, D)
    #     expert_outputs = torch.stack([self.experts[i](dispatched_input[i]) for i in range(self.num_experts)])
    #
    #     # 转置回来 (E, B*N, D) -> (B*N, E, D)
    #     expert_outputs = expert_outputs.transpose(0, 1)
    #
    #     # --- 2.4 组合专家输出 ---
    #     # 用同样的分发权重来组合专家的输出
    #     sparse_output_flat = torch.einsum('be,bed->bd', combine_weights, expert_outputs)
    #
    #     # 恢复形状
    #     sparse_output = sparse_output_flat.view(batch_size, num_tokens, dim)
    #
    #     # ==================================================
    #     # === 2. Super Expert Fusion (方案 B 的核心) ===
    #     # ==================================================
    #     # 将稀疏专家的组合输出送入超级专家进行信息融合
    #     # fused_output = self.super_expert_fusion_layer(sparse_output)
    #     # sparse_output = fused_output
    #
    #     # =====================================
    #     # === 3. Final Combination (最终合并) ===
    #     # =====================================
    #     final_output = shared_output + sparse_output
    #
    #     # =====================================
    #     # === 4. Load Balancing Loss (修正版) ===
    #     # =====================================
    #     load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    #     if self.training and self.load_balancing_alpha > 0:
    #         # Switch Transformer中提出的标准负载均衡损失
    #         # router_probs: 在所有可选专家上的概率分布
    #         # 使用router_logits_flat，它已经包含了mask的效果
    #         router_probs = F.softmax(router_logits_flat, dim=-1)
    #
    #         # expert_load: 每个专家实际处理的token比例 (E,)
    #         # F.one_hot -> (B*N, top_k, E) -> sum(dim=1) -> (B*N, E) -> mean(dim=0) -> (E,)
    #         expert_load = (dispatch_weights > 0).float().mean(dim=0)
    #
    #         # Loss = N * sum(f_i * P_i)
    #         # expert_load 和 router_probs 对于被mask掉的专家都将接近0，所以计算是正确的
    #         load_balancing_loss = self.load_balancing_alpha * self.num_experts * torch.sum(
    #             expert_load * router_probs.mean(dim=0))
    #
    #     # --- 【核心修改】根据开关决定返回内容 ---
    #     if return_expert_outputs:
    #         # 在可视化模式下，返回最终输出以及所有专家的中间输出
    #         # 我们返回一个字典，这样更清晰
    #         return final_output, load_balancing_loss, top_k_indices, expert_outputs
    #
    #
    #     if not self.training:
    #         return final_output, load_balancing_loss, top_k_indices, None
    #     return final_output, load_balancing_loss, None, None

    # def forward(self, x, condition_embedding, is_vfm_condition=False):
    #     # x: 输入特征, 形状 (B, N, D) (batch_size, num_tokens, dim)
    #     # condition_embedding: 条件嵌入, 形状 (B, cond_D) (batch_size, condition_dim)
    #     batch_size, num_tokens, dim = x.shape
    #     # ============================================
    #     # === 1. Shared Expert Path (共享专家路径) ===
    #     # ============================================
    #     # 所有输入token都直接通过共享专家
    #     shared_output = self.shared_expert(x)  # -> (B, N, D)
    #
    #     # ============================================
    #     # === 2. Sparse Experts Path (稀疏专家路径) ===
    #     # ============================================
    #     x_flat = x.reshape(-1, dim)  # 展平为 (B*N, D)
    #
    #     # 获取路由权重，注意这里传递的是原始的 (B, cond_D) condition_embedding
    #     # ConditionedRouter内部会处理其扩展
    #     routing_weights = self.router(x_flat, condition_embedding, is_vfm_condition=is_vfm_condition)  # (B*N, num_experts)
    #
    #     # 选择top_k个Expert
    #     top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    #     gating_weights = F.softmax(top_k_logits, dim=-1)  # -> (B*N, top_k)
    #     # 归一化top_k权重 (如果top_k > 1)
    #     if self.top_k > 1:
    #         top_k_weights_norm = top_k_weights / (torch.sum(top_k_weights, dim=-1, keepdim=True) + 1e-6)
    #     else:
    #         top_k_weights_norm = top_k_weights
    #
    #     # 初始化最终输出
    #     sparse_output_flat = torch.zeros_like(x_flat)
    #     flat_token_indices = torch.arange(x_flat.size(0), device=x.device)  # 扁平化的token索引
    #
    #     # 分发token到Expert并计算输出
    #     for k_idx in range(self.top_k):  # 遍历top_k的选择
    #         current_expert_indices_for_k = top_k_indices[:, k_idx]  # 当前第k选择的Expert索引
    #         current_weights_for_k = top_k_weights_norm[:, k_idx]  # 当前第k选择的权重
    #
    #         for i in range(self.num_experts):  # 遍历所有Expert
    #             mask_expert_i = (current_expert_indices_for_k == i)  # 找到分配给Expert i的token
    #             tokens_routed_to_expert_i_indices = flat_token_indices[mask_expert_i]
    #
    #             if tokens_routed_to_expert_i_indices.numel() > 0:  # 如果有token分配给此Expert
    #                 tokens_to_process_by_expert_i = x_flat[tokens_routed_to_expert_i_indices]
    #                 expert_output = self.experts[i](tokens_to_process_by_expert_i)  # Expert处理
    #                 # 用门控权重加权Expert的输出
    #                 weighted_expert_output = expert_output * current_weights_for_k[tokens_routed_to_expert_i_indices].unsqueeze(1)
    #                 # 将加权输出加到最终结果 (scatter操作)
    #                 sparse_output_flat.index_add_(0, tokens_routed_to_expert_i_indices, weighted_expert_output)
    #
    #     sparse_output = sparse_output_flat.reshape(batch_size, num_tokens, dim)  # 恢复形状
    #
    #     # =====================================
    #     # === 3. Final Combination (最终合并) ===
    #     # =====================================
    #     # 【新增】将共享专家的输出和稀疏专家的输出相加
    #     final_output = shared_output + sparse_output
    #
    #     # 计算负载均衡损失 (辅助损失)
    #     load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    #     if self.training and self.load_balancing_alpha > 0:
    #         # P_i: 每个Expert被选中的平均概率 (来自路由器的直接输出)
    #         P_i = routing_weights.mean(dim=0)
    #         # f_i: 每个Expert实际处理的token的比例 (基于top_1选择)
    #         f_i_counts = torch.histc(top_k_indices[:, 0].float(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    #         f_i = f_i_counts / (f_i_counts.sum() + 1e-6)
    #         load_balancing_loss = self.load_balancing_alpha * self.num_experts * torch.sum(f_i * P_i)
    #
    #     # 蒸馏损失不在此处计算，将根据导师的思路在外部基于模型整体的条件化输出进行
    #     return final_output, load_balancing_loss
    def _compute_dynamic_weight(self, current_val, target_val):
        """
        DeepSeekMoE 的核心 Trick: 动态计算权重
        Weight = 10 ^ round(log10(Target / Current))
        """
        if current_val < 1e-9:  # 防止除零
            return torch.tensor(0.0, device=current_val.device)

        ratio = target_val / current_val
        # log10 -> round -> power of 10
        weight = 10 ** torch.round(torch.log10(ratio))
        return weight

    def compute_ortho_loss(self, expert_outputs):
        """
        [新增] 使用矩阵运算计算 Top-K 专家输出的正交损失。
        Args:
            expert_outputs: [Batch*Seq, TopK, Hidden_Dim]
                            注意：这里必须是 gather 之后的数据，确保同一行是同一个Token的不同TopK专家
        """
        if self.top_k <= 1:
            return torch.tensor(0.0, device=expert_outputs.device)

        # 1. 归一化 (只看方向，不看长度)
        norm_outputs = F.normalize(expert_outputs, p=2, dim=-1, eps=1e-6)

        # 2. 计算 TopK 内部两两相似度
        # [N, K, D] @ [N, D, K] -> [N, K, K]
        sim_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

        # 3. 目标：非对角线元素为 0 (即相互垂直)
        identity = torch.eye(self.top_k, device=expert_outputs.device).unsqueeze(0)

        # 只惩罚非对角线部分
        ortho_loss = torch.sum((sim_matrix - identity) ** 2)

        # 4. 归一化 Loss 幅度
        count = expert_outputs.size(0) * (self.top_k * (self.top_k - 1))
        return ortho_loss / (count + 1e-6)

    def compute_weight_ortho_loss(self):
        loss = 0
        # 遍历每个组
        for i in range(self.router.num_groups):
            start = i * self.num_experts_per_group
            end = start + self.num_experts_per_group

            # 拿到这一组 K 个专家的 Up-projection 权重
            # weights: (K, D_out, D_in) -> Flatten -> (K, P)
            group_weights = torch.stack([e.fc2.weight for e in self.experts[start:end]])
            flat_weights = group_weights.view(self.num_experts_per_group, -1)

            # 归一化
            norm_weights = F.normalize(flat_weights, p=2, dim=1)

            # 算 Gram Matrix
            gram = torch.mm(norm_weights, norm_weights.t())

            # 减去单位阵 (希望非对角线为0)
            eye = torch.eye(self.num_experts_per_group, device=gram.device)
            loss += torch.sum((gram - eye) ** 2)

        return loss

    def _compute_weight_ortho_loss(self):
        """
        计算专家组内部的参数权重正交损失。
        """
        # 如果权重为0或专家数少于2，则不计算
        if self.ortho_loss_weight <= 0 or self.num_experts_per_group <= 1:
            return torch.tensor(0.0, device=self.experts[0].fc1.weight.device)

        total_ortho_loss = 0.0
        num_groups = self.num_experts // self.num_experts_per_group

        for g_id in range(num_groups):
            start_idx = g_id * self.num_experts_per_group
            end_idx = start_idx + self.num_experts_per_group

            # 1. 提取当前组所有专家的 fc1.weight
            # 并将每个 (H, D) 的权重矩阵展平为向量
            w_list = [
                self.experts[i].fc1.weight.view(1, -1)
                for i in range(start_idx, end_idx)
            ]

            # 2. 将向量列表堆叠成一个矩阵 (K, P), K=组内专家数, P=参数量
            w_matrix = torch.cat(w_list, dim=0)

            # 3. 对每个专家的参数向量进行 L2 归一化
            w_norm = F.normalize(w_matrix, p=2, dim=1)

            # 4. 计算 Gram 矩阵 (相似度矩阵): W_norm @ W_norm^T
            # 结果是一个 (K, K) 的矩阵
            gram_matrix = torch.mm(w_norm, w_norm.t())

            # 5. 计算与单位矩阵的差值
            # 我们希望 Gram 矩阵是一个单位矩阵 (对角线为1，其余为0)
            identity = torch.eye(self.num_experts_per_group, device=gram_matrix.device)

            # Loss = ||Gram - I||^2，惩罚所有非对角线元素和偏离1的对角线元素
            loss_group = torch.mean((gram_matrix - identity) ** 2)

            total_ortho_loss += loss_group

        # 返回所有组的平均损失
        return total_ortho_loss / num_groups

    def init_expert_from_base_mlp(self, expert: nn.Module, base_mlp: nn.Module):
        """
        用 base_block.mlp 初始化 expert。
        - 如果维度一致，直接 load_state_dict
        - 如果 hidden dim 变小（如 4x -> 2x），则截取前面的通道做初始化
        - 如果 hidden dim 变大，则把已有权重拷进去，其余随机初始化
        """
        with torch.no_grad():
            base_sd = base_mlp.state_dict()
            exp_sd = expert.state_dict()

            # 完全一致：直接加载
            same_shape = True
            for k in exp_sd:
                if k not in base_sd or exp_sd[k].shape != base_sd[k].shape:
                    same_shape = False
                    break

            if same_shape:
                expert.load_state_dict(base_sd)
                return

            # 否则做 shape-aware 拷贝
            new_sd = {}
            for k, v_exp in exp_sd.items():
                if k not in base_sd:
                    new_sd[k] = v_exp
                    continue

                v_base = base_sd[k]

                if v_base.shape == v_exp.shape:
                    new_sd[k] = v_base
                    continue

                # Linear weight: 2D tensor
                if v_base.ndim == 2 and v_exp.ndim == 2:
                    out_dim = min(v_base.shape[0], v_exp.shape[0])
                    in_dim = min(v_base.shape[1], v_exp.shape[1])

                    v_new = v_exp.clone()
                    v_new[:out_dim, :in_dim] = v_base[:out_dim, :in_dim]
                    new_sd[k] = v_new

                # Bias / LayerNorm-like: 1D tensor
                elif v_base.ndim == 1 and v_exp.ndim == 1:
                    dim = min(v_base.shape[0], v_exp.shape[0])
                    v_new = v_exp.clone()
                    v_new[:dim] = v_base[:dim]
                    new_sd[k] = v_new

                else:
                    # 其他情况保留 expert 自身初始化
                    new_sd[k] = v_exp

            expert.load_state_dict(new_sd, strict=False)
    # @staticmethod
    # def initialize_expert_from_base_block(expert, base_mlp, verbose=False):
    #     """
    #     将 base_mlp (标准FFN) 的权重加载到 expert (带Norm和Scale的FFN) 中，
    #     并正确初始化新增的层。
    #
    #     Args:
    #         expert: 你的 UniversalExpert 实例
    #         base_mlp: 原始 ViT block 中的 mlp 模块
    #     """
    #     # 1. 获取基础模型的权重 (使用 deepcopy 防止内存共享)
    #     base_state_dict = copy.deepcopy(base_mlp.state_dict())
    #
    #     # 2. 加载权重 (strict=False)
    #     # 这会自动匹配 fc1.weight, fc1.bias, fc2.weight, fc2.bias 等同名参数
    #     # 同时会忽略 output_norm 和 output_scale，因为 base_state_dict 里没有
    #     missing_keys, unexpected_keys = expert.load_state_dict(base_state_dict, strict=False)
    #
    #     if verbose:
    #         print(f"Expert Init - Missing keys (expected): {missing_keys}")
    #         # missing_keys 应该包含 'output_norm.weight', 'output_norm.bias', 'output_scale'
    #
    #     # 3. 【关键】手动初始化新增的 Output Scale
    #     # 初始化为 1.0，保证初始状态下 scale 不改变数值量级
    #     if hasattr(expert, 'output_scale') and expert.output_scale is not None:
    #         nn.init.constant_(expert.output_scale, 1.0)
    #
    #     # 4. 【关键】手动初始化新增的 Output LayerNorm
    #     # 按照 Identity 初始化：Weight=1, Bias=0
    #     # 这样初始时刻 Norm 层不会剧烈扭曲 Pre-trained Linear 层的输出方向
    #     if hasattr(expert, 'output_norm') and expert.output_norm is not None:
    #         nn.init.constant_(expert.output_norm.weight, 1.0)
    #         nn.init.constant_(expert.output_norm.bias, 0.0)
    #
    #     # 5. 可选：重置 Dropout (通常不需要，因为 Dropout 没有权重)
    #     return expert

# class ConditionedMoETransformerBlock(nn.Module):
#     """条件化的Transformer块，其FFN部分是ConditionedMoELayer"""
#
#     def __init__(self, input_dim, num_heads, condition_dim, num_moe_experts, moe_top_k=1,
#                  mlp_ratio=4.0,  # 标准FFN的MLP比例，这里可能不直接用，因为MoE有自己的expert_hidden_ratio
#                  expert_hidden_ratio=4.0,  # MoE Expert内部FFN的隐层比例
#                  qkv_bias=True, noisy_gating=True,
#                  attn_drop=0., proj_drop=0.,  # dropout率
#                  base_block=None,
#                  router_type="no_structure_router",
#                  task_training=False,
#                  ):
#         super().__init__()
#
#         global_print("Initializing ConditionedMoETransformerBlock with MoE FFN.")
#         # --- 1. 直接从instance_block克隆注意力子层 ---
#         # 确保注意力部分的行为与标准块完全一致
#         self.norm1 = copy.deepcopy(base_block.norm1)
#         self.attn = copy.deepcopy(base_block.attn)
#         self.ls1 = copy.deepcopy(base_block.ls1)
#         self.drop_path1 = copy.deepcopy(base_block.drop_path1)
#
#         # --- 2. FFN/MoE 子层 ---
#         # 克隆标准块的 norm2, ls2, 和 drop_path2，以保证结构同构
#         self.norm2 = copy.deepcopy(base_block.norm2)
#         self.ls2 = copy.deepcopy(base_block.ls2)
#         self.drop_path2 = copy.deepcopy(base_block.drop_path2)
#
#         # 核心替换：用你的MoE层替换标准的Mlp层
#         # 注意：我们不再需要 instance_block.mlp
#         self.moe_ffn_layer = ConditionedMoELayer(
#             input_dim=input_dim,  # 从实例块获取维度
#             num_experts=num_moe_experts,
#             condition_dim=condition_dim,
#             top_k=moe_top_k,
#             expert_hidden_ratio=expert_hidden_ratio,
#             noisy_gating=noisy_gating,
#             router_type=router_type,
#             base_block=base_block
#         )
#
#         # self.moe_ffn_layer = ConditionedMoELayer(input_dim, num_moe_experts, condition_dim, top_k=moe_top_k,
#         #                                          expert_hidden_ratio=expert_hidden_ratio, noisy_gating=noisy_gating)
#
#
#
#     def forward(self, x, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None, task_id=None):
#         # x: 输入特征 (B,N,D)
#         # condition_embedding: 条件嵌入 (B, cond_D)
#
#         # 自注意力部分
#         # normed_x = self.norm1(x)
#         # attn_output, _ = self.attn(query=normed_x, key=normed_x, value=normed_x, need_weights=False)
#         # x_attn = x + self.drop_path(attn_output) # 如果使用DropPath
#         # x_attn = x + attn_output
#         # --- 注意力部分 (与标准Block完全相同) ---
#         x_attn = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         # MoE FFN部分
#         moe_input = self.norm2(x_attn)
#         moe_output, aux_loss, top_k_indices = self.moe_ffn_layer(moe_input,
#                                                   condition_embedding,
#                                                   is_vfm_condition=is_vfm_condition,
#                                                   vfm_teacher_id=vfm_teacher_id,
#                                                   task_id=task_id)
#         # x_ffn = x_attn + self.drop_path(moe_output) # 如果使用DropPath
#         # x_ffn = x_attn + moe_output
#         x_ffn = x_attn + self.drop_path2(self.ls2(moe_output))
#         if not self.training:
#             # 返回最终输出，以及用于分析的内部状态
#             return x_ffn, aux_loss, None, top_k_indices.detach(), None
#         return x_ffn, aux_loss, None, None, None

class ConditionedMoETransformerBlock(nn.Module):
    """
    条件化的Transformer块，其FFN部分是ConditionedMoELayer。
    此版本经过修改，与GatedMoETransformerBlock的接口兼容。
    """

    def __init__(self,
                 input_dim,
                 num_heads,
                 condition_dim,
                 num_moe_experts,
                 moe_top_k=3,
                 mlp_ratio=4.0,
                 expert_hidden_ratio=4.0,
                 qkv_bias=True, noisy_gating=True,
                 attn_drop=0., proj_drop=0.,
                 base_block=None,
                 router_type="no_structure_router",
                 task_training=False,  # task_training 似乎没有被使用，但为了兼容性保留
                 num_tasks=0,  # 添加 num_tasks 以匹配 GatedMoETransformerBlock 的接口
                 ):
        super().__init__()

        global_print("Initializing ConditionedMoETransformerBlock (API-Compatible Version).")
        # --- 1. 直接从 base_block 克隆注意力子层 ---
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.attn = copy.deepcopy(base_block.attn)
        self.ls1 = copy.deepcopy(base_block.ls1)
        self.drop_path1 = copy.deepcopy(base_block.drop_path1)

        # --- 2. FFN/MoE 子层 ---
        self.norm2 = copy.deepcopy(base_block.norm2)
        self.ls2 = copy.deepcopy(base_block.ls2)
        self.drop_path2 = copy.deepcopy(base_block.drop_path2)

        # 核心：用 ConditionedMoELayer 替换标准的 Mlp 层
        self.moe_ffn_layer = ConditionedMoELayer(
            input_dim=input_dim,
            num_experts=num_moe_experts,
            condition_dim=condition_dim,
            top_k=moe_top_k,
            expert_hidden_ratio=expert_hidden_ratio,
            noisy_gating=noisy_gating,
            router_type=router_type,
            base_block=base_block,
            num_tasks=num_tasks  # 将 num_tasks 传递给 MoE 层
        )

    # 修改 forward 方法的签名和返回值以匹配 GatedMoETransformerBlock
    def forward(self, x: torch.Tensor, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None,
                return_expert_outputs=False, task_id=None):
        # x: 输入特征 (B,N,D)
        # condition_embedding: 条件嵌入 (B, cond_D)

        # --- 注意力部分 (与标准Block完全相同) ---
        x_attn = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        # --- MoE FFN 部分 ---
        moe_input = self.norm2(x_attn)

        # 调用MoE层，确保它能处理 return_expert_outputs 参数
        # 假设 ConditionedMoELayer 返回 (output, aux_loss, top_k_indices, all_expert_outputs)
        moe_output, aux_loss, top_k_indices, all_expert_outputs = self.moe_ffn_layer(
            moe_input,
            condition_embedding,
            is_vfm_condition=is_vfm_condition,
            vfm_teacher_id=vfm_teacher_id,
            return_expert_outputs=return_expert_outputs,
            task_id=task_id
        )

        x_ffn = x_attn + self.drop_path2(self.ls2(moe_output))

        # --- 构造与 GatedMoETransformerBlock 兼容的返回值 ---
        # 由于没有门控，我们需要创建占位符

        # 1. 门控值 (g): 因为完全使用MoE路径，等效于门控值为0
        g_placeholder = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        # 2. 门控损失 (gate_loss): 没有门控，损失为0
        gate_loss_placeholder = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 3. 门控预激活值 (gate_pre_activation): 设为负无穷，因为 sigmoid(-inf) = 0
        pre_act_placeholder = torch.full_like(g_placeholder, float('-inf'))

        # 根据不同模式返回
        if return_expert_outputs:
            # 这里的 top_k_indices 可能是 None，如果 MoE 层在训练时不返回它，需要处理
            top_k_indices_detached = top_k_indices.detach() if top_k_indices is not None else None
            return x_ffn, aux_loss, g_placeholder.detach(), top_k_indices_detached, all_expert_outputs

        if not self.training:
            top_k_indices_detached = top_k_indices.detach() if top_k_indices is not None else None
            return x_ffn, aux_loss, g_placeholder.detach(), top_k_indices_detached, None

        # 训练时返回完整的元组
        return x_ffn, aux_loss, g_placeholder, gate_loss_placeholder, pre_act_placeholder
class GatedMoETransformerBlock(nn.Module):
    """
    一个与你的标准Block架构完全兼容的混合MoE Transformer块。
    通过传入一个预训练的block实例来初始化其标准部分。
    """

    def __init__(
            self,
            input_dim: int,
            # --- 传入一个预训练好的Block实例 ---
            base_block: nn.Module,
            # --- MoE特定参数 ---
            condition_dim: int,
            num_moe_experts: int,
            moe_top_k: int,
            expert_hidden_ratio: float = 4.0,
            noisy_gating: bool = True,
            router_type="no_structure_router",
            gate_min_val: float = 0.5,  # 门控输出的最小值
            gate_max_val: float = 1.0,  # 门控输出的最大值
            num_tasks=0,
            block_idx=-1
    ) -> None:
        super().__init__()

        global_print("Initializing GatedMoETransformerBlock with Gated MoE FFN.")
        # --- 1. 直接从base_block克隆注意力子层 ---
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.attn = copy.deepcopy(base_block.attn)
        self.ls1 = copy.deepcopy(base_block.ls1)
        self.drop_path1 = copy.deepcopy(base_block.drop_path1)

        # --- 2. FFN/MoE 子层 ---
        self.norm2 = copy.deepcopy(base_block.norm2)

        # 路径 A: 标准FFN/MLP路径，直接从base_block克隆
        # 注意：我们将 base_block.mlp 克隆到 self.standard_ffn
        self.standard_ffn = copy.deepcopy(base_block.mlp)

        # 路径 B: 你的条件化MoE层 (这是新创建的模块)
        self.moe_ffn_layer = ConditionedMoELayer(
            input_dim=input_dim,  # 从base_block获取维度信息
            num_experts=num_moe_experts,
            condition_dim=condition_dim,
            top_k=moe_top_k,
            expert_hidden_ratio=expert_hidden_ratio,
            noisy_gating=noisy_gating,
            router_type=router_type,
            base_block=base_block,
            num_tasks=num_tasks
        )

        self.gate_linear = nn.Linear(input_dim, 1)
        # self.task_layer_gate = nn.Parameter(torch.zeros(num_tasks, 1))

        self.gate_activation = ConstrainedSigmoid(min_val=gate_min_val, max_val=gate_max_val) #(min_val=gate_min_val, max_val=gate_max_val)
        with torch.no_grad():
            self.gate_linear.bias.fill_(2.0)

        # self.spatial_gate_residual =None


        # self.block_idx = block_idx
        # if self.block_idx in [8, 11]:
        #     self.spatial_gate_residual = SpatialGateResidual(input_dim)
        # else:
        #     self.spatial_gate_residual = None


        # self.block_idx = block_idx
        # if self.block_idx in [8, 11]:
        #     learn_task_floor = True
        #     min_floor_limit = 0.30  # 深层允许 gate 最低到 0.30
        # else:
        #     learn_task_floor = False
        #     min_floor_limit = 0.50  # 浅层其实不会用到
        #
        # self.gate = TaskAwareGate(
        #     input_dim=input_dim,
        #     num_tasks=num_tasks,  # stage 2 的任务数
        #     gate_max_val=1.0,  # 你也可以设成原来的 gate_max_val
        #     fixed_min_val=0.5,
        #     learn_task_floor=learn_task_floor,
        #     min_floor_limit=min_floor_limit,
        #     delta_scale=0.10,
        #     block_idx=block_idx
        # )
        # 克隆 LayerScale 和 DropPath
        self.ls2 = copy.deepcopy(base_block.ls2)
        self.drop_path2 = copy.deepcopy(base_block.drop_path2)


    def forward(self, x: torch.Tensor, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None, return_expert_outputs=False, task_id=None):

        # print(f"------------------Block---{self.block_idx}------------------------")
        # print("task_gate:", self.gate.task_gate.data.squeeze())
        # if self.gate.task_floor_logits is not None:
        #     print("task_floor_logits:", self.gate.task_floor_logits.data.squeeze())
        #
        # print("task_gate:", self.gate.task_gate.data.squeeze())
        # if self.gate.task_floor_logits is not None:
        #     print("task_floor_logits:", self.gate.task_floor_logits.data.squeeze())

        # --- 注意力部分 (与标准Block完全相同) ---
        x_attn = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        # --- 门控专家路径融合 ---
        ffn_input = self.norm2(x_attn)



        # 计算两条路径的输出
        standard_output = self.standard_ffn(ffn_input)


        moe_output, aux_loss, top_k_indices, all_expert_outputs = self.moe_ffn_layer(
            ffn_input,
            condition_embedding,
            is_vfm_condition=is_vfm_condition,
            vfm_teacher_id=vfm_teacher_id,
            return_expert_outputs=return_expert_outputs,
            task_id=task_id
        )

        # ==================== 诊断代码 ====================
        # 计算两个输出向量之间的余弦相似度
        # 我们在最后一个维度 (特征维度) 上计算
        # F.cosine_similarity 会返回一个形状为 (B, N) 的张量
        # similarity = F.cosine_similarity(standard_output, moe_output, dim=-1)
        # 打印一个 batch 中所有 token 的平均相似度
        # print(f"Layer X - Path Similarity: {similarity.mean().item()}")
        # 计算门控值并进行融合
        # g = self.gate(ffn_input)
        # 获取预激活值和门控值

        gate_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # gate_pre_activation = self.gate_linear(ffn_input)
        # g = self.gate_activation(gate_pre_activation)

        gate_pre_activation = self.gate_linear(ffn_input)  # [B, N, 1]

        # if self.spatial_gate_residual is not None:
        #     _, _, H, W = self.tokens_to_2d(ffn_input,has_cls_token=True)
        #     spatial_score = self.spatial_gate_residual(ffn_input, H, W)
        #     gate_pre_activation = gate_pre_activation - spatial_score

        g = self.gate_activation(gate_pre_activation)


        # # tid = task_id
        # # print("selected task_gate:", self.gate.task_gate[tid].item())
        # # if self.gate.task_floor_logits is not None:
        # #     print("selected floor_logit:", self.gate.task_floor_logits[tid].item())
        # g, gate_pre_activation = self.gate(ffn_input, task_ids=task_id)  # [B,N,1]



        ################# for ContextAwareGate ##########################
        # g, gate_pre_activation = self.gate(ffn_input)

        # ############# for smarterGate ###############################
        # # ==================== CLS Token 适配 SmarterGate 的关键逻辑 ====================
        # # 1. 分离 CLS token 和 patch tokens
        # # ffn_input 形状: (B, 1 + N, C)
        # cls_token_in = ffn_input[:, 0:1, :]  # 形状: (B, 1, C)
        # patch_tokens_in = ffn_input[:, 1:, :]  # 形状: (B, N, C)
        #
        # # 2. 只将 patch tokens 送入 SmarterGate (self.gate 现在是 SmarterGate 实例)
        # #    它会返回门控值和预激活值
        # g_patches, pre_act_patches = self.gate(patch_tokens_in)  # 形状: (B, N, 1), (B, N, 1)
        #
        # # 3. 为 CLS token 单独创建门控值和预激活值
        # #    我们通常希望 CLS token 的 FFN 路径永远是开启的，所以 g_cls = 1.0。
        # #    为了得到 g_cls=1.0，预激活值需要是一个非常大的正数。
        # g_cls = torch.ones_like(g_patches[:, 0:1, :])  # 形状: (B, 1, 1)，值全为1
        #
        # # 使用一个较大的值来确保激活后接近1，或直接用inf
        # pre_act_cls = torch.full_like(g_cls, float('inf'))  # 形状: (B, 1, 1)
        #
        # # 4. 将 CLS 和 patch 的结果拼接起来，恢复原始序列顺序
        # g = torch.cat([g_cls, g_patches], dim=1)  # 形状: (B, 1 + N, 1)
        # gate_pre_activation = torch.cat([pre_act_cls, pre_act_patches], dim=1)  # 形状: (B, 1 + N, 1)
        #
        # # ==============================================================================
        # M_samples = x.shape[1]  # 使用token数量作为采样数
        # gate_loss = self.gate_diversity_loss_sampled(ffn_input, g, num_samples=M_samples)
        # ###########################################################################################

        fused_output = g * standard_output + (1.0 - g) * moe_output
        # fused_output = moe_output
        # fused_output = standard_output
        # aux_loss = aux_loss * 0.0  # 不使用MoE路径，仅使用标准FFN路径时，辅助损失为0
        # --- 应用LayerScale, DropPath并添加残差连接 (与标准Block完全相同) ---
        x_ffn = x_attn + self.drop_path2(self.ls2(fused_output))

        # # 在评估模式下，可以返回门控值和aux_loss以供分析
        # if not self.training:
        #     return x_ffn, aux_loss, g.detach()

        # print(gate_pre_activation)
        # print(g)

        if return_expert_outputs:
            return x_ffn, aux_loss, g.detach(), top_k_indices.detach(), all_expert_outputs

        if not self.training:
            # 返回最终输出，以及用于分析的内部状态
            return x_ffn, aux_loss, g.detach(), top_k_indices.detach(), None

        return x_ffn, aux_loss, g, gate_loss, gate_pre_activation

    def tokens_to_2d(self, x, has_cls_token=False):
        """
        x: [B, N, C]
        return:
            x_patch: [B, Np, C]
            x2d: [B, C, H, W]
            H, W
        """
        B, N, C = x.shape

        if has_cls_token:
            x_patch = x[:, 1:, :]
        else:
            x_patch = x

        Np = x_patch.shape[1]
        H = W = int(Np ** 0.5)
        assert H * W == Np, f"Token number {Np} is not a square."

        x2d = x_patch.transpose(1, 2).reshape(B, C, H, W)
        return x_patch, x2d, H, W

    def gate_diversity_loss_sampled(self, ffn_input: torch.Tensor, g: torch.Tensor, num_samples: int):
        """
        计算基于随机配对采样的门控多样性损失。

        Args:
            ffn_input (torch.Tensor): FFN的输入, 形状 (B, N, C)。
            g (torch.Tensor): 门控值, 形状 (B, N, 1)。
            num_samples (int): 每个样本要采样的token对的数量。

        Returns:
            torch.Tensor: 计算出的损失值。
        """
        B, N, C = ffn_input.shape
        if N <= 1:
            return 0.0

        # 确保采样数不超过可能的最大对数
        M = min(num_samples, N * (N - 1))

        # 归一化特征以方便计算余弦相似度
        ffn_input_norm = F.normalize(ffn_input, p=2, dim=-1,eps=1e-6)  # 形状: (B, N, C)

        # 随机生成 M 对索引
        # 生成索引 i
        rand_indices_i = torch.randint(0, N, (B, M), device=ffn_input.device)
        # 生成索引 j，并确保 i != j
        rand_indices_j = torch.randint(0, N, (B, M), device=ffn_input.device)
        # 如果 i 和 j 相同，将 j 加 1 再取模，这是一种简单避免 i=j 的方法
        same_indices = (rand_indices_i == rand_indices_j)
        rand_indices_j[same_indices] = (rand_indices_j[same_indices] + 1) % N

        # 使用 gather 提取采样的 token 特征和门控值
        # gather 需要的 index 形状需要与源张量对齐
        idx_i = rand_indices_i.unsqueeze(-1).expand(-1, -1, C)  # 形状: (B, M, C)
        idx_j = rand_indices_j.unsqueeze(-1).expand(-1, -1, C)  # 形状: (B, M, C)

        tokens_i = torch.gather(ffn_input_norm, 1, idx_i)  # 形状: (B, M, C)
        tokens_j = torch.gather(ffn_input_norm, 1, idx_j)  # 形状: (B, M, C)

        g_i = torch.gather(g, 1, rand_indices_i.unsqueeze(-1))  # 形状: (B, M, 1)
        g_j = torch.gather(g, 1, rand_indices_j.unsqueeze(-1))  # 形状: (B, M, 1)

        # 计算采样对的余弦相似度 S (向量化操作)
        # (B, M, C) * (B, M, C) -> sum -> (B, M)
        S_pairs = (tokens_i * tokens_j).sum(dim=-1)

        # 计算采样对的门控值差异 D
        D_pairs = (g_i - g_j).squeeze(-1).pow(2)  # 形状: (B, M)

        # 计算对比损失
        # 我们希望 S_pairs 高时 D_pairs 低，S_pairs 低时 D_pairs 高
        # 最小化 S * D，最大化 (1-S) * D (等价于最小化 -(1-S)*D)
        loss = (S_pairs * D_pairs - (1 - S_pairs) * D_pairs).mean()

        return loss
# --- 主干网络和顶层模型 ---
class Condition_MoE_ViT(nn.Module):
    """集成了条件化MoE层的Vision Transformer主干"""

    def __init__(
            self,
            vit_base_model,
            condition_dim,
            num_moe_experts,
            moe_top_k=1,
            expert_hidden_ratio=4.0,
            moe_layers_indices=None,
            noisy_gating=True,
            out_indices=None,  # 任务特定的输出索引
            moe_type='conditioned',  # 'conditioned' 或 'gated'
            router_type="no_structure_router",
            gate_constraint_ranges=[[0.8, 1.0],[0.6, 1.0],[0.4, 1.0],[0.2, 1.0]],  # 仅在moe_type='gated'时使用
            num_tasks=0
            ):
        # vit_base_model: 一个预训练或基础的ViT模型实例，用于获取参数和非MoE层
        # condition_dim: 条件嵌入维度
        # num_moe_experts: MoE层中的Expert数量
        # moe_layers_indices: 一个列表，指定哪些Transformer Block的索引应该替换为MoE Block
        super().__init__()
        self.embed_dim = vit_base_model.embed_dim
        # 从基础模型复制Patch Embedding, Positional Embedding等组件
        self.patch_embed = copy.deepcopy(vit_base_model.patch_embed)
        self.pos_embed = copy.deepcopy(vit_base_model.pos_embed)  # 注意：确保这个方法或属性名与您的基础ViT一致
        self.num_classes = vit_base_model.num_classes
        self.global_pool = vit_base_model.global_pool
        self.num_features = self.head_hidden_size = self.embed_dim

        self.num_reg_tokens = vit_base_model.num_reg_tokens
        self.has_class_token = vit_base_model.has_class_token
        self.no_embed_class = vit_base_model.no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = vit_base_model.dynamic_img_size
        self.grad_checkpointing = vit_base_model.grad_checkpointing

        if hasattr(vit_base_model, 'cls_token') and vit_base_model.cls_token is not None:
            self.cls_token = copy.deepcopy(vit_base_model.cls_token)
        else:
            self.cls_token = None
        self.pos_drop = copy.deepcopy(getattr(vit_base_model, 'pos_drop', nn.Identity()))
        self.patch_drop = copy.deepcopy(getattr(vit_base_model, 'patch_drop', nn.Identity()))
        self.norm_pre = copy.deepcopy(getattr(vit_base_model, 'norm_pre', nn.Identity()))
        self.cls_token = copy.deepcopy(getattr(vit_base_model, 'cls_token', nn.Identity()))
        self.reg_token = copy.deepcopy(getattr(vit_base_model, 'reg_token', None))  # Register token (如果有)
        self.pos_drop = copy.deepcopy(getattr(vit_base_model, 'pos_drop', nn.Identity()))
        self.patch_drop = copy.deepcopy(getattr(vit_base_model, 'patch_drop', nn.Identity()))

        self.out_indices = out_indices
        # self.no_embed_class = getattr(vit_base_model, 'no_embed_class', False)  # 是否不嵌入CLS token

        # CLS token 和 Register token 的处理:
        # 它们会作为普通token通过patch_embed和pos_embed。
        # num_prefix_tokens 通常包括CLS token，可能还包括Register tokens。
        # 它们会正常参与后续的Attention和MoE层的计算。
        self.num_prefix_tokens = getattr(vit_base_model, 'num_prefix_tokens', 1 if hasattr(vit_base_model, 'cls_token') else 0)

        self.blocks = nn.ModuleList()
        depth = len(vit_base_model.blocks)
        # 假设所有标准块的注意力头数一致，且MoE块也使用相同的头数
        num_heads = vit_base_model.blocks[0].attn.num_heads

        assert len(moe_layers_indices) == len(gate_constraint_ranges)
        moe_config = dict(zip(moe_layers_indices, gate_constraint_ranges))
        for i in range(depth):
            if moe_layers_indices and i in moe_layers_indices:  # 如果当前层是MoE层
                if moe_type == 'conditioned':
                    self.blocks.append(ConditionedMoETransformerBlock(input_dim=self.embed_dim,
                                                                      num_heads=num_heads,
                                                                      condition_dim=condition_dim,
                                                                      num_moe_experts=num_moe_experts, moe_top_k=moe_top_k,
                                                                      expert_hidden_ratio=expert_hidden_ratio,
                                                                      noisy_gating=noisy_gating,
                                                                      base_block=vit_base_model.blocks[i],
                                                                      router_type=router_type,
                                                                      num_tasks=num_tasks,
                                                                      ))
                elif moe_type == 'gated':
                    # print(moe_config[i])
                    gate_min_val, gate_max_val = moe_config[i]
                    print(f"Layer {i}: Gated MoE with gate constraints [{gate_min_val}, {gate_max_val}]")
                    self.blocks.append(GatedMoETransformerBlock(input_dim=self.embed_dim,
                                                                condition_dim=condition_dim,
                                                                num_moe_experts=num_moe_experts, moe_top_k=moe_top_k,
                                                                expert_hidden_ratio=expert_hidden_ratio,
                                                                noisy_gating=noisy_gating,
                                                                base_block=vit_base_model.blocks[i],
                                                                router_type=router_type,
                                                                gate_min_val=gate_min_val,
                                                                gate_max_val=gate_max_val,
                                                                num_tasks=num_tasks,
                                                                block_idx=i
                                                                ))
                else:
                    raise ValueError(f"Unsupported moe_type: {moe_type}. Choose 'conditioned' or 'gated'.")
            else:  # 标准Transformer块
                self.blocks.append(copy.deepcopy(vit_base_model.blocks[i]))

        self.norm = copy.deepcopy(vit_base_model.norm) if hasattr(vit_base_model, 'norm') and vit_base_model.norm is not None else nn.Identity()  # ViT末尾的最终LayerNorm (如果存在)

    def forward(self, x, condition_embedding, is_vfm_condition=False,vfm_teacher_id=None):
        # x: 输入图像 (B,C,H,W)
        # condition_embedding: 条件嵌入 (B, cond_D)
        x = self.patch_embed(x)  # -> (B, N_patches, D_embed)
        x = self._pos_embed(x)  # -> (B, N_total_tokens, D_embed), N_total_tokens = N_patches + num_prefix_tokens
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        total_aux_loss_agg = torch.tensor(0.0, device=x.device, dtype=x.dtype)  # 累积MoE辅助损失
        gate_regularization_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        gate_list = []  # 用于存储门控值（如果需要）
        topk_indices_list = []  # 用于存储top_k_indices（如果需要）
        all_gate_pre_activations = []

        for i, blk in enumerate(self.blocks):
            pre_act_val = None  # 预先定义，避免作用域问题

            if not isinstance(blk, Block):  # 如果是MoE块
                # 统一调用MoE块，并接收所有可能的返回值
                x, aux_loss, g, top_k_indices, pre_act_val = blk(x,
                                                                 condition_embedding,
                                                                 is_vfm_condition=is_vfm_condition,
                                                                 vfm_teacher_id=vfm_teacher_id)

                # 累加辅助损失
                total_aux_loss_agg += aux_loss

                # 统一收集门控值（如果存在）
                if g is not None:
                    gate_list.append(g)

                # 在评估模式下收集 top_k_indices
                if not self.training:
                    topk_indices_list.append(top_k_indices)

                # 在训练模式下收集 pre-activations
                if self.training:
                    all_gate_pre_activations.append(pre_act_val)

            else:  # 标准ViT块
                x = blk(x)

        # if not self.training:
        #
        #     for i, blk in enumerate(self.blocks):
        #         if not isinstance(blk, Block):  # 如果是MoE块
        #             x, aux_loss, g, top_k_indices,_ = blk(x, condition_embedding, is_vfm_condition=is_vfm_condition, vfm_teacher_id=vfm_teacher_id)  # 需要传递条件嵌入
        #             total_aux_loss_agg += aux_loss
        #             gate_list.append(g)
        #             topk_indices_list.append(top_k_indices)
        #         else:  # 标准ViT块
        #             x = blk(x)  # 标准块不接收条件嵌入
        #
        # else:
        #
        #     for i, blk in enumerate(self.blocks):
        #
        #         if not isinstance(blk, Block):  # 如果是MoE块
        #             x, aux_loss, g, top_k_indices, pre_act = blk(x, condition_embedding, is_vfm_condition=is_vfm_condition,vfm_teacher_id=vfm_teacher_id )  # 需要传递条件嵌入
        #             total_aux_loss_agg += aux_loss
        #             gate_list.append(g)
        #             all_gate_pre_activations.append(pre_act)
        #         else:  # 标准ViT块
        #             x = blk(x)  # 标准块不接收条件嵌入

        if self.norm is not None:  # 应用最终的归一化层
            x = self.norm(x)

        if all_gate_pre_activations != [] and all_gate_pre_activations[0] is not None:
            for pre_act in all_gate_pre_activations:
                # L2 正则化，鼓励预激活值靠近0，从而保持在Sigmoid的动态区
                gate_regularization_loss += torch.mean(pre_act ** 2)
        # 返回最终的token序列特征和累积的MoE辅助损失
        # CLS token通常是 x[:, 0]
        # Register tokens (如果有) 也在x中，但一般不直接用于最终输出
        return x, total_aux_loss_agg, gate_list, topk_indices_list, gate_regularization_loss

    def forward_intermediate_features(self, x, condition_embedding, is_vfm_condition=False, vfm_teacher_id=None, return_expert_outputs=False, task_id=None):
        # x: 输入图像 (B,C,H,W)
        # condition_embedding: 条件嵌入 (B, cond_D)
        x = self.patch_embed(x)  # -> (B, N_patches, D_embed)
        x = self._pos_embed(x)  # -> (B, N_total_tokens, D_embed), N_total_tokens = N_patches + num_prefix_tokens
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        total_aux_loss_agg = torch.tensor(0.0, device=x.device, dtype=x.dtype)  # 累积MoE辅助损失
        gate_regularization_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        output = []  # 用于收集中间特征
        gate_list = []  # 用于存储门控值（如果需要）
        topk_indices_list = []  # 用于存储top_k_indices（如果需要）
        all_gate_pre_activations = []

        # 单一的、统一的循环
        moe_blk_index=0
        for i, blk in enumerate(self.blocks):
            pre_act_val = None  # 预先定义，避免作用域问题

            if not isinstance(blk, Block):  # MoE Block
                # if is_vfm_condition:
                #     # Stage 1: condition_embedding 是 (B, D)
                #     cond = condition_embedding
                # else:
                #     # Stage 2: condition_embedding 是 (B, L, D)
                #     # 我们取当前 MoE 层的 embedding: (B, D)
                #     cond = condition_embedding[:, moe_blk_index, :]
                #     moe_blk_index += 1

                cond = condition_embedding
                x, aux_loss, g, gate_loss_or_top_k_indices, pre_act_val = blk(x,
                                                                 cond,
                                                                 is_vfm_condition=is_vfm_condition,
                                                                 vfm_teacher_id=vfm_teacher_id,
                                                                 return_expert_outputs=return_expert_outputs,
                                                                 task_id=task_id
                                                                 )

                # 累加辅助损失
                total_aux_loss_agg += aux_loss

                # 统一收集门控值（如果存在）
                # print(g)
                if g is not None:
                    gate_list.append(g)

                # 在评估模式下收集 top_k_indices
                if not self.training:
                    topk_indices_list.append(gate_loss_or_top_k_indices)

                # 在训练模式下收集 pre-activations
                if self.training:
                    all_gate_pre_activations.append(pre_act_val)

            else:  # 标准ViT块
                x = blk(x)

            # 统一处理中间特征的输出
            if i in self.out_indices:
                if self.norm is not None:
                    x_norm = self.norm(x)
                else:
                    x_norm = x
                output.append(x_norm)


        # if not self.training:
        #     for i, blk in enumerate(self.blocks):
        #         if not isinstance(blk, Block):  # 如果是MoE块
        #             x, aux_loss, g, top_k_indices, _ = blk(x,
        #                                               condition_embedding,
        #                                               is_vfm_condition=is_vfm_condition,
        #                                               vfm_teacher_id=vfm_teacher_id)  # 需要传递条件嵌入
        #             total_aux_loss_agg += aux_loss
        #             if g is not None:
        #                 gate_list.append(g)
        #                 topk_indices_list.append(top_k_indices)
        #         else:  # 标准ViT块
        #             x = blk(x)  # 标准块不接收条件嵌入
        #
        #         if i in self.out_indices:
        #             if self.norm is not None:  # 应用最终的归一化层
        #                 x_norm = self.norm(x)
        #             else:
        #                 x_norm = x
        #             output.append(x_norm)  # 收集中间特征
        # else:
        #     for i, blk in enumerate(self.blocks):
        #         if not isinstance(blk, Block):  # 如果是MoE块
        #             x, aux_loss, g, top_k_indices, pre_act = blk(x,
        #                                                 condition_embedding,
        #                                                 is_vfm_condition=is_vfm_condition,
        #                                                 vfm_teacher_id=vfm_teacher_id)  # 需要传递条件嵌入
        #             total_aux_loss_agg += aux_loss
        #             all_gate_pre_activations.append(pre_act)
        #             gate_list.append(g)
        #         else:  # 标准ViT块
        #             x = blk(x)  # 标准块不接收条件嵌入
        #
        #         if i in self.out_indices:
        #             if self.norm is not None:  # 应用最终的归一化层
        #                 x_norm = self.norm(x)
        #             else:
        #                 x_norm = x
        #             output.append(x_norm)  # 收集中间特征


        # if all_gate_pre_activations != [] and all_gate_pre_activations[0] is not None:
        #     for pre_act in all_gate_pre_activations:
        #         # L2 正则化，鼓励预激活值靠近0，从而保持在Sigmoid的动态区
        #         gate_regularization_loss += torch.mean(pre_act ** 2)

        gate_regularization_loss += gate_loss_or_top_k_indices if self.training else 0.0
        # 返回最终的token序列特征和累积的MoE辅助损失
        # CLS token通常是 x[:, 0]
        # Register tokens (如果有) 也在x中，但一般不直接用于最终输出
        return output, total_aux_loss_agg, gate_list, topk_indices_list, gate_regularization_loss

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
            if self.pos_embed is None:
                return x.view(x.shape[0], -1, x.shape[-1])

            if self.dynamic_img_size:
                B, H, W, C = x.shape
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
                x = x.view(B, -1, C)
            else:
                pos_embed = self.pos_embed

            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

            if self.no_embed_class:
                # deit-3, updated JAX (big vision)
                # position embedding does not overlap with class token, add then concat
                x = x + pos_embed
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
            else:
                # original timm, JAX, and deit vit impl
                # pos_embed has entry for class token, concat then add
                if to_cat:

                    x = torch.cat(to_cat + [x], dim=1)
                x = x + pos_embed

            return self.pos_drop(x)





# 假设你已经有了标准的Transformer Block实现，比如从timm库中
# from timm.models.vision_transformer import Block
# 如果没有，你需要一个标准的Block实现
# class Block(nn.Module): ...
class Fat_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        base_block=None,
    ) -> None:
        super().__init__()
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.attn = copy.deepcopy(base_block.attn)
        self.ls1 = copy.deepcopy(base_block.ls1)
        self.drop_path1 = copy.deepcopy(base_block.drop_path1)
        self.norm2 = copy.deepcopy(base_block.norm2)
        # 使用更“胖”的MLP层
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = copy.deepcopy(base_block.ls2)
        self.drop_path2 = copy.deepcopy(base_block.drop_path2)

        self._init_fat_mlp(base_block.mlp, self.mlp)
        # self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_norm=qk_norm,
        #     attn_drop=attn_drop,
        #     proj_drop=proj_drop,
        #     norm_layer=norm_layer,
        # )
        # self.ls1 = (
        #     LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # )
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        #
        # self.norm2 = norm_layer(dim)
        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        # self.ls2 = (
        #     LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # )
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _init_fat_mlp(self, src_mlp, dst_mlp):
        """
        src_mlp: 原始 Base 模型的 MLP (小)
        dst_mlp: 新建的 Fat 模型的 MLP (大)
        """
        # 获取源权重
        w1 = src_mlp.fc1.weight.data
        b1 = src_mlp.fc1.bias.data
        w2 = src_mlp.fc2.weight.data
        b2 = src_mlp.fc2.bias.data

        # 获取目标形状
        dst_h_dim = dst_mlp.fc1.out_features
        src_h_dim = src_mlp.fc1.out_features

        # 计算倍数 (如果是5倍, repeat=5)
        # 注意：这里假设整除，如果不能整除稍微复杂点，但通常倍数是整数
        repeat = dst_h_dim // src_h_dim
        if dst_h_dim % src_h_dim != 0:
            print("Warning: Fat MLP width is not an integer multiple. Initialization might be suboptimal.")

        with torch.no_grad():
            # --- 处理 fc1 (升维层) ---
            # w1 shape: [hidden, in_dim] -> 在 dim 0 (hidden) 上复制
            dst_mlp.fc1.weight.data.copy_(w1.repeat(repeat, 1))
            # b1 shape: [hidden] -> 复制
            dst_mlp.fc1.bias.data.copy_(b1.repeat(repeat))

            # --- 处理 fc2 (降维层) ---
            # w2 shape: [out_dim, hidden] -> 在 dim 1 (hidden) 上复制
            # 重要：必须要除以 repeat，保证输出数值范围不变
            dst_mlp.fc2.weight.data.copy_(w2.repeat(1, repeat) / repeat)

            # b2 shape: [out_dim] -> 不需要复制，直接拷贝即可，bias和hidden维度无关
            dst_mlp.fc2.bias.data.copy_(b2)

        print(f"Fat MLP initialized from Base MLP using Net2Net tiling (Repeat: {repeat}x).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class Fat_ViT(nn.Module):
    """
    一个非MoE的Vision Transformer，但在特定层级使用了更“胖”的FFN层，
    以在参数量上匹配对应的MoE ViT。
    这是一个用于架构对比实验的关键模型。
    """

    def __init__(
            self,
            vit_base_model,
            fat_mlp_ratio=4.0,  # “胖”层的MLP ratio
            fat_layers_indices=None,  # 指定哪些层需要变“胖”
            out_indices=None,
    ):
        # vit_base_model: 一个预训练或基础的ViT模型实例
        # fat_mlp_ratio: “胖”FFN层的MLP扩展比例
        # fat_layers_indices: 需要变胖的层的索引列表
        super().__init__()

        # --- 这部分代码与你的 Condition_MoE_ViT 完全相同，用于复制基础结构 ---
        self.embed_dim = vit_base_model.embed_dim
        self.patch_embed = copy.deepcopy(vit_base_model.patch_embed)
        self.pos_embed = copy.deepcopy(vit_base_model.pos_embed)
        self.num_classes = vit_base_model.num_classes
        self.global_pool = vit_base_model.global_pool
        self.num_features = self.head_hidden_size = self.embed_dim
        self.num_reg_tokens = vit_base_model.num_reg_tokens
        self.has_class_token = vit_base_model.has_class_token
        self.no_embed_class = vit_base_model.no_embed_class
        self.dynamic_img_size = vit_base_model.dynamic_img_size
        self.grad_checkpointing = vit_base_model.grad_checkpointing
        self.cls_token = copy.deepcopy(getattr(vit_base_model, 'cls_token', None))
        self.pos_drop = copy.deepcopy(getattr(vit_base_model, 'pos_drop', nn.Identity()))
        self.patch_drop = copy.deepcopy(getattr(vit_base_model, 'patch_drop', nn.Identity()))
        self.norm_pre = copy.deepcopy(getattr(vit_base_model, 'norm_pre', nn.Identity()))
        self.reg_token = copy.deepcopy(getattr(vit_base_model, 'reg_token', None))
        self.num_prefix_tokens = getattr(vit_base_model, 'num_prefix_tokens',
                                         1 if hasattr(vit_base_model, 'cls_token') else 0)
        self.out_indices = out_indices
        # --- 基础结构复制结束 ---

        self.blocks = nn.ModuleList()
        depth = len(vit_base_model.blocks)

        for i in range(depth):
            # 获取基础Block的配置，以便复用
            base_block = vit_base_model.blocks[i]
            num_heads = base_block.attn.num_heads
            # 假设Block接受mlp_ratio作为参数
            original_mlp_ratio = base_block.mlp.mlp_ratio if hasattr(base_block.mlp, 'mlp_ratio') else 4.0

            # --- 核心改动 ---
            if fat_layers_indices and i in fat_layers_indices:
                # 如果当前层是“胖”层，我们使用 fat_mlp_ratio
                print(f"Creating a FAT block at index {i} with mlp_ratio={fat_mlp_ratio}")

                self.blocks.append(
                    Fat_Block(
                        dim=self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=fat_mlp_ratio,  # 使用更大的MLP ratio
                        base_block=base_block,
                        # ... 确保传入其他Block需要的参数, e.g., qkv_bias, drop_path, etc.
                    )
                )
            else:
                # 标准Transformer块，直接从基础模型复制或用原始ratio创建
                print(f"Creating a REGULAR block at index {i} with mlp_ratio={original_mlp_ratio}")
                self.blocks.append(copy.deepcopy(base_block))

        self.norm = copy.deepcopy(vit_base_model.norm) if hasattr(vit_base_model, 'norm') and vit_base_model.norm is not None else nn.Identity()

    # --- `forward` 和 `forward_intermediate_features` ---
    # 这两个函数变得非常简单，因为它们不再需要处理MoE的特殊逻辑和条件嵌入

    def forward(self, x, condition_embedding=None, is_vfm_condition=True, vfm_teacher_id=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        # 注意：这里不再有辅助损失
        return x, torch.tensor(0.0, device=x.device, dtype=x.dtype), None, None, torch.tensor(0.0, device=x.device, dtype=x.dtype)


    def forward_intermediate_features(self, x, condition_embedding=None, is_vfm_condition = False, vfm_teacher_id = None, return_expert_outputs = False, task_id = None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # 在timm的实现中，通常是在block之后提取特征，norm在最后
                # 但你的MoE模型在提取特征时加了norm，为了公平对比，这里也加上
                # 如果你的标准ViT是在最后才norm，这里应该去掉norm
                feature_out = self.norm(x) if self.norm is not None else x
                output.append(feature_out)

        # 注意：这里不再有辅助损失
        return output, torch.tensor(0.0, device=x.device, dtype=x.dtype), None, None, torch.tensor(0.0, device=x.device, dtype=x.dtype)


    # _pos_embed 函数与你的 Condition_MoE_ViT 完全相同，直接复制即可
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
            if self.pos_embed is None:
                return x.view(x.shape[0], -1, x.shape[-1])

            if self.dynamic_img_size:
                B, H, W, C = x.shape
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
                x = x.view(B, -1, C)
            else:
                pos_embed = self.pos_embed

            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

            if self.no_embed_class:
                # deit-3, updated JAX (big vision)
                # position embedding does not overlap with class token, add then concat
                x = x + pos_embed
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
            else:
                # original timm, JAX, and deit vit impl
                # pos_embed has entry for class token, concat then add
                if to_cat:

                    x = torch.cat(to_cat + [x], dim=1)
                x = x + pos_embed

            return self.pos_drop(x)

class Condition_MoE_PRISM(nn.Module):
    """顶层学生模型，封装了条件化MoE ViT主干和各种头"""

    def __init__(
            self,
            img_size,
            vit_name,
            vit_pretrained: bool = None,
            tea_dims: dict = None,
            tasks: list = None,
            tasks_dict: dict = None,  # 例如: {"task_0": 0, "task_1": 1}
            condition_dim = None,
            num_moe_experts = 15,
            moe_top_k=9,
            moe_layers_indices=None,
            noisy_gating=True,
            task_head_configs: dict = None,  # 例如: {"task_0": {"out_features": 10}, "task_1": {"out_features": 5}}
            vfm_projection_configs: dict = None, # 例如: {"vfm_0": {"out_features": 768}, "vfm_1": {"out_features": 512}}
            freeze_vit: bool = False,
            freeze: bool = False,
            lora_config: dict = None,

            *args,
            **kwargs,
                 ):
        super().__init__()

        if vit_pretrained is None:
            env_value = os.environ.get("PRISM_VIT_PRETRAINED")
            vit_pretrained = True if env_value is None else env_value.lower() in {"1", "true", "on", "yes"}

        # 1. 加载基础ViT模型 (作为模板和参数来源)
        if vit_name == "vit_small":
            from models.backbones.vit import vit_small_patch16_384

            vit_base_model_instance = vit_small_patch16_384(img_size=img_size, pretrained=vit_pretrained)
            self.out_indices = kwargs["out_indices_cfg_for_task"]["small"]  # 任务特定的输出索引
        elif vit_name == "vit_base":
            from models.backbones.vit import vit_base_patch16_384

            vit_base_model_instance = vit_base_patch16_384(img_size=img_size, pretrained=vit_pretrained)
            self.out_indices = kwargs["out_indices_cfg_for_task"]["base"]  # 任务特定的输出索引
        elif vit_name == "vit_large":
            from models.backbones.vit import vit_large_patch16_384

            vit_base_model_instance = vit_large_patch16_384(img_size=img_size, pretrained=vit_pretrained)
            self.out_indices = kwargs["out_indices_cfg_for_task"]["large"]
        else:
            raise NotImplementedError("vit_name not supported")

        if kwargs["vit_checkpoint_path"] is not None:
            global_print("Loading ViT checkpoint from {}".format(kwargs["vit_checkpoint_path"]))
            state_dict = torch.load(kwargs["vit_checkpoint_path"], map_location='cpu')
            state_dict = {k.replace("vit.", ""): v for k, v in state_dict.items()}
            # print(checkpoint)
            # 可能需要处理 checkpoint 字典的键名以匹配 vit_base_model_instance 的 state_dict
            vit_base_model_instance.load_state_dict(state_dict, strict=False)


        self.embed_dim = vit_base_model_instance.embed_dim  # 学生主干的特征维度
        self.fea_size = (img_size[0] // 16, img_size[1] // 16)
        if condition_dim is None:
            self.condition_dim = vit_base_model_instance.embed_dim  # 默认使用主干的特征维度作为条件嵌入维度
        else:
            self.condition_dim = condition_dim

        # 2. 条件嵌入 (可学习)
        # num_tasks = len(tasks_dict) # 如果要训一个统一的多任务model
        # self.tasks_dict = tasks_dict  # 任务名称到condition ID的映射
        num_tasks = len(tasks) if tasks else 0  # 如果tasks参数提供了任务列表，则使用其长度
        num_vfm_teachers = len(tea_dims) if tea_dims else 0  # VFM教师数量
        self.tasks = tasks
        self.num_tasks = num_tasks
        self.num_vfm_teachers = num_vfm_teachers
        # 为每个任务ID创建一个嵌入向量
        self.task_condition_embeddings = nn.Embedding(num_tasks, self.condition_dim)
        self.task_composer_weights = nn.Parameter(torch.randn(num_tasks, num_vfm_teachers) * 0.02)
        # self.num_moe_layers = len(self.out_indices)
        # self.task_composer_weights = nn.Parameter(
        #     torch.randn(self.num_moe_layers, num_tasks, num_vfm_teachers) * 0.02
        # )
        # 为每个VFM教师ID创建一个嵌入向量
        self.vfm_condition_embeddings = nn.Embedding(num_vfm_teachers, self.condition_dim)
        # 假设 Edge 是 task_id=3
        # prior_weights = torch.zeros(num_tasks, 3)
        #
        # # 对于 Edge 任务，给 SAM (idx 2) 一个巨大的初始分，比如 5.0 (Softmax后接近 0.99)
        # # 给 DINO (idx 0) 一个小的辅助分，比如 1.0
        # prior_weights[3] = torch.tensor([1.0, 0.0, 5.0])
        #
        # # 对于 Semseg 任务，给 DINO 一个大分
        # prior_weights[0] = torch.tensor([5.0, 1.0, 0.0])
        #
        # # 赋值给 composer
        # with torch.no_grad():
        #     self.task_composer_weights.copy_(prior_weights)
        if self.task_composer_weights.shape[0] == 5:
            with torch.no_grad():
                self.task_composer_weights.fill_(0.0)

                # 让 Edge 任务 (Task 3) 对 SAM (Idx 2) 的偏好达到极致
                # [ -10, -10, 20 ] -> Softmax 后几乎是纯 SAM

                # 让 Semseg (Task 0) 对 DINO (Idx 0) 的偏好达到极致
                self.task_composer_weights[0] = torch.tensor([2.0, -1.0, -1.0])
                self.task_composer_weights[1] = torch.tensor([2.0, -1.0, -1.0])
                self.task_composer_weights[2] = torch.tensor([-1.0, -1.0, 2.0])
                self.task_composer_weights[3] = torch.tensor([-1.0, -1.0, 2.0])
                self.task_composer_weights[4] = torch.tensor([-1.0, 2.0, -1.0])

        elif self.task_composer_weights.shape[0] == 4:
            with torch.no_grad():
                self.task_composer_weights.fill_(0.0)

                # 让 Edge 任务 (Task 3) 对 SAM (Idx 2) 的偏好达到极致
                # [ -10, -10, 20 ] -> Softmax 后几乎是纯 SAM

                # 让 Semseg (Task 0) 对 DINO (Idx 0) 的偏好达到极致
                self.task_composer_weights[0] = torch.tensor([2.0, -1.0, -1.0])
                self.task_composer_weights[1] = torch.tensor([-1.0, -1.0, 2.0])
                self.task_composer_weights[2] = torch.tensor([-1.0, -1.0, 2.0])
                self.task_composer_weights[3] = torch.tensor([2.0, -1.0, -1.0])
                # # 遍历每一层进行初始化
                # for l in range(self.num_moe_layers):
                #     # Semseg (Task 0) -> DINO (Idx 0) 偏好
                #     # self.task_composer_weights[l, 0] = torch.tensor([2.0, -1.0, -1.0])
                #     # # Human Parts? (Task 1) -> SAM (Idx 2) 偏好 (根据你的代码逻辑)
                #     # self.task_composer_weights[l, 1] = torch.tensor([-1.0, -1.0, 2.0])
                #     # # Normals? (Task 2) -> SAM (Idx 2) 偏好
                #     # self.task_composer_weights[l, 2] = torch.tensor([-1.0, -1.0, 2.0])
                #     # # Edge (Task 3) -> DINO (Idx 0) 偏好
                #     # self.task_composer_weights[l, 3] = torch.tensor([2.0, -1.0, -1.0])
                #
                #     self.task_composer_weights[l, 0] = torch.tensor([1.0, 1.0, 1.0])
                #     # Human Parts? (Task 1) -> SAM (Idx 2) 偏好 (根据你的代码逻辑)
                #     self.task_composer_weights[l, 1] = torch.tensor([1.0, 1.0, 1.0])
                #     # Normals? (Task 2) -> SAM (Idx 2) 偏好
                #     self.task_composer_weights[l, 2] = torch.tensor([1.0, 1.0, 1.0])
                #     # Edge (Task 3) -> DINO (Idx 0) 偏好
                #     self.task_composer_weights[l, 3] = torch.tensor([1.0, 1.0, 1.0])

        self.task_free_condition_embeddings = nn.Embedding(1, self.condition_dim)

        self.vfm_p_drop = kwargs.get("vfm_p_drop", 0.3)
        # self.vfm_teacher_free_condition_embeddings = nn.Embedding(1, self.condition_dim)

        # self.backbone = vit_base_model_instance
        # 3. 实例化 Condition_MoE_ViT 主干网络
        # self.backbone = Fat_ViT(
        #     vit_base_model=vit_base_model_instance,
        #     fat_mlp_ratio=20.0,      # 计算得出: moe_top_k * expert_hidden_ratio = 4 * 4.0
        #     fat_layers_indices=moe_layers_indices,
        #     out_indices=self.out_indices,  # 任务特定的输出索引
        # )
        vit_type = kwargs["vit_type"]
        if vit_type == "moh":  # 如果指定使用 Mixture of Heads
            global_print(f"Initializing MoH (Mixture-of-Heads) Backbone with Top-K={moe_top_k}")

            self.backbone = MoH_ViT(
                vit_base_model=vit_base_model_instance,
                num_selected_heads=moe_top_k,  # 复用 moe_top_k 参数作为 head 的 top-k
                moh_layers_indices=moe_layers_indices,
                noisy_gating=noisy_gating,
                out_indices=self.out_indices
            )
        elif vit_type == "conditioned_moe":
            self.backbone = Condition_MoE_ViT(
                vit_base_model=vit_base_model_instance,
                condition_dim=self.condition_dim,
                num_moe_experts=num_moe_experts,
                moe_top_k=moe_top_k,
                expert_hidden_ratio=kwargs["expert_hidden_ratio"],
                moe_layers_indices=moe_layers_indices,
                noisy_gating=noisy_gating,
                out_indices=self.out_indices,  # 任务特定的输出索引
                moe_type=kwargs["moe_type"],
                router_type=kwargs["router_type"],
                gate_constraint_ranges=kwargs["gate_constraint_ranges"], #[[0.8, 1.0], [0.6, 1.0], [0.4, 1.0], [0.2, 1.0]]
                num_tasks=self.num_tasks
                # expert_hidden_ratio可以从vit_base_model推断或作为参数传入
            )
        elif vit_type == "fat_vit":
            self.backbone = Fat_ViT(
                vit_base_model=vit_base_model_instance,
                fat_mlp_ratio=20.0,      # 计算得出: moe_top_k * expert_hidden_ratio = 4 * 4.0
                fat_layers_indices=moe_layers_indices,
                out_indices=self.out_indices,  # 任务特定的输出索引
            )
        else:
            raise NotImplementedError("moe_type not supported")

        del vit_base_model_instance


        torch.cuda.empty_cache()
        if freeze_vit:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif lora_config:
            from peft import LoraConfig, get_peft_model

            self.backbone = get_peft_model(self.backbone, LoraConfig(**lora_config))
            if tasks:
                global_print("Full fine-tune ViT and LoRA in stage 2")
                for param in self.backbone.parameters():
                    param.requires_grad = True
            self.backbone.print_trainable_parameters()
        # 4. M个任务特定的输出头
        self.task_heads = None
        # self.task_heads = nn.ModuleDict()
        # if task_head_configs:
        #     for task_name_idx_str, config in task_head_configs.items():  # task_name_idx_str "0", "1", ...
        #         self.task_heads[task_name_idx_str] = nn.Linear(self.embed_dim, config["out_features"])

        # 5. N个VFM教师蒸馏用的投影头 (用于阶段1)
        # # 将主干的通用输出特征投影到与每个VFM教师输出维度匹配的空间
        # self.vfm_projection_heads = nn.ModuleDict()
        # if vfm_projection_configs:
        #     for teacher_name_idx_str, config in vfm_projection_configs.items():  # teacher_name_idx_str "0", "1", ...
        #         self.vfm_projection_heads[teacher_name_idx_str] = nn.Linear(self.embed_dim, config["out_features"])

        self.vfm_projection_heads = nn.ModuleDict()  # 用于存储每个VFM教师的投影头


        for tea_no in tea_dims.keys():
            if tea_dims[tea_no] == 0:
                break
            p_type = kwargs["vfm_projector_type"]
            if p_type == "tp":
                global_print("################# VFM Projector type is TP ##################")
                ############# tp projector #############
                for l_ind in range(len(self.out_indices)):
                    self.vfm_projection_heads[tea_no + "_" +str(l_ind)] = build_projector(
                        input_dim=self.embed_dim,
                        output_dim=tea_dims[tea_no],
                        extra_args=None
                    )
            elif p_type == "mlp":
                global_print("################# VFM Projector type is MLP ##################")
                ############# mlp projector #############
                for l_ind in range(len(self.out_indices)):
                    # 实例化 MLP
                    self.vfm_projection_heads[tea_no + "_" + str(l_ind)] = MlpProjector(
                        input_dim=self.embed_dim,
                        output_dim=tea_dims[tea_no],
                        mlp_ratio=1.0  # 这里的ratio可以根据需要调整，通常1.0或4.0
                    )
            else:
                global_print("################# VFM Projector type is linear ##################")
                ############ linear projector #############
                for l_ind in range(len(self.out_indices)):
                    self.vfm_projection_heads[tea_no + "_" +str(l_ind)] = nn.Linear(self.embed_dim, tea_dims[tea_no])
        self._init_weights_custom()  # 初始化新添加的层

    def _init_weights_custom(self, m=None):
        # 递归或直接调用apply来初始化
        if m is None:
            if hasattr(self, 'task_condition_embeddings'): self.task_condition_embeddings.apply(
                self._init_weights_custom)
            if hasattr(self, 'vfm_condition_embeddings'): self.vfm_condition_embeddings.apply(
                self._init_weights_custom)
            if hasattr(self, 'task_free_condition_embeddings'): self.task_free_condition_embeddings.apply(
                self._init_weights_custom)
            if hasattr(self, 'vfm_free_condition_embeddings'): self.vfm_free_condition_embeddings.apply(
                self._init_weights_custom)
            if hasattr(self, 'task_heads') and self.task_heads: self.task_heads.apply(self._init_weights_custom)
            if hasattr(self, 'vfm_projection_heads') and self.vfm_projection_heads: self.vfm_projection_heads.apply(
                self._init_weights_custom)
            return

        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):  # 初始化嵌入层
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_task_condition_embedding(self, task_ids=None):
        # task_ids: (B,) 长整型张量, 或None
        # vfm_teacher_ids: (B,) 长整型张量, 或None
        # 在一个给定的阶段，通常只有一种ID是激活的（或它们的组合）
        if task_ids is not None:
            # B = task_ids.shape[0]
            # unconditional_embedding_vector = self.vfm_teacher_condition_embeddings.weight.mean(dim=0)
            # unconditional_embedding_vector = self.vfm_teacher_free_condition_embeddings.weight
            return self.task_condition_embeddings(task_ids) # + unconditional_embedding_vector.expand(B,-1)  # (B, condition_dim)
        # elif vfm_teacher_ids is not None:
        #     B = vfm_teacher_ids.shape[0]
        #     unconditional_embedding_vector = self.task_free_condition_embeddings.weight
        #     return self.vfm_condition_embeddings(vfm_teacher_ids) + unconditional_embedding_vector.expand(B,-1) # (B, condition_dim)
        else:
            # 必须提供一种条件ID，否则无法进行条件化路由
            # unconditional_embedding_vector = self.task_free_condition_embeddings.weight
            # return unconditional_embedding_vector
            raise ValueError("必须提供 task_ids 作为条件。")

    # def get_task_condition_embedding(self, task_ids=None):
    #     # 1. 获取混合权重 (B, 3)
    #     # task_id: (B,)
    #     weights = self.task_composer_weights[task_ids]  # (B, 3)
    #
    #     # 2. Softmax 归一化，确保总能量守恒
    #     # attention_scores = F.softmax(weights, dim=-1)  # (B, 3)
    #     # 在 get_task_embedding 中
    #     temperature = 2.0  # 温度越高，分布越平滑；温度越低，越尖锐
    #     attention_scores = F.softmax(weights / temperature, dim=-1)
    #
    #     # 3. 获取所有 VFM Embeddings (1, 3, Dim)
    #     all_vfm_embs = self.vfm_condition_embeddings.weight.unsqueeze(0)
    #
    #     # 4. 加权求和 (Attention)
    #     # (B, 3, 1) * (1, 3, Dim) -> (B, 3, Dim) -> sum -> (B, Dim)
    #     mixed_embedding = (attention_scores.unsqueeze(-1) * all_vfm_embs).sum(dim=1)
    #     # 检查 Embedding 是否真的变了
    #     # if self.training and random.random() < 0.01:
    #     #     print(f"Current Task ID: {task_ids[0]}")
    #     #     print(f"Condition Emb Mean: {mixed_embedding.mean().item()}")
    #         # 如果 Semseg 和 Edge 的 Emb Mean 一样，说明 Composer 没起作用
    #     # global_print(self.task_composer_weights)
    #     return mixed_embedding

    # def get_task_condition_embedding(self, task_ids=None):
    #     """
    #     返回: (B, num_moe_layers, dim)
    #     """
    #     # task_ids: (B,)
    #     # self.task_composer_weights: (L, T, V)  L=Layers, T=Tasks, V=VFMs
    #
    #     # 1. 索引获取当前 Batch 的权重
    #     # 使用高级索引：选取所有层 (:), 指定的任务 (task_ids), 所有VFM (:)
    #     # 结果形状: (num_moe_layers, Batch, num_vfms) -> (L, B, V)
    #     weights = self.task_composer_weights[:, task_ids, :]
    #
    #     # 2. Softmax 归一化 (在 VFM 维度上)
    #     temperature = 2.0
    #     attention_scores = F.softmax(weights / temperature, dim=-1)  # (L, B, V)
    #
    #     # 3. 获取所有 VFM Embeddings
    #     # self.vfm_condition_embeddings.weight: (V, D)
    #     all_vfm_embs = self.vfm_condition_embeddings.weight  # (V, D)
    #
    #     # 4. 加权求和 (Einstein Summation)
    #     # L: Layers, B: Batch, V: VFMs, D: Dim
    #     # attention_scores (L, B, V) * all_vfm_embs (V, D) -> (L, B, D)
    #     mixed_embedding_layers = torch.einsum('lbv, vd -> lbd', attention_scores, all_vfm_embs)
    #
    #     # 5. 转置为 (B, L, D) 以方便在 forward 中按 batch 迭代
    #     mixed_embedding_layers = mixed_embedding_layers.transpose(0, 1).contiguous()
    #
    #     # 调试打印 (可选，打印第一层和最后一层的均值，检查是否分化)
    #     # if self.training and random.random() < 0.001:
    #     #     global_print(f"L0 Weights: {attention_scores[0, 0].detach().cpu().numpy()}")
    #     #     global_print(f"L{self.num_moe_layers-1} Weights: {attention_scores[-1, 0].detach().cpu().numpy()}")
    #     global_print(self.task_composer_weights)
    #     return mixed_embedding_layers

    def get_vfm_condition_embedding(self, vfm_teacher_ids, training=False):
        """
        获取VFM条件嵌入，并在训练时对批次中的每个样本按概率p进行随机丢弃。
        被丢弃的样本将使用一个可学习的"free"嵌入来替代。
        """

        return self.vfm_condition_embeddings(vfm_teacher_ids)

        ########### for 随机丢弃+门控机制 ############
        # # --- 安全检查 ---
        # if vfm_teacher_ids is None or training is False:
        #     # print("vfm_teacher_ids not provided or training is False.")
        #     B = vfm_teacher_ids.shape[0]
        #     free_embedding_expanded = self.task_free_condition_embeddings.weight.expand(B, -1)
        #     return free_embedding_expanded
        # # 只有在训练模式下才进行丢弃
        # if training:
        #     # --- 核心改动：样本级别丢弃 ---
        #     B = vfm_teacher_ids.shape[0]  # 获取批次大小
        #
        #     # 1. 为批次中的每个样本生成一个随机数，决定是否丢弃
        #     # drop_mask 是一个布尔张量，形状为 (B,)，True表示需要丢弃
        #     drop_mask = torch.rand(B, device=vfm_teacher_ids.device) < self.vfm_p_drop
        #
        #     # 2. 首先，正常查询所有样本的VFM ID嵌入
        #     all_embeddings = self.vfm_condition_embeddings(vfm_teacher_ids)
        #
        #     # 3. 将 "free" 嵌入扩展到与批次对齐的形状
        #     free_embedding_expanded = self.task_free_condition_embeddings.weight.expand(B, -1)
        #
        #     # 4. 使用 torch.where 根据 drop_mask 进行选择
        #     # 如果 drop_mask[i] 为 True，则选择 free_embedding
        #     # 否则，选择正常的 all_embeddings[i]
        #     # unsqueeze(1) 是为了让(B,)的mask可以对(B, D)的tensor进行广播
        #     final_embeddings = torch.where(
        #         drop_mask.unsqueeze(1),
        #         free_embedding_expanded,
        #         all_embeddings
        #     )
        #
        #     return final_embeddings
        # else:
        #     # 在非训练模式（验证/推理）下，永远不要丢弃
        #     return self.vfm_condition_embeddings(vfm_teacher_ids)


    def forward(self, batch, vfm_training=True, task_training=False):
        # images: 输入图像 (B, C, H, W)
        # task_ids: 当前批次对应的任务ID (B,)，用于阶段2
        # vfm_teacher_ids: 当前批次对应的VFM教师ID (B,)，用于阶段1

        # 从 batch 中解析 images 和 vfm_teacher_ids (或 task_ids)


        if isinstance(batch, list) or isinstance(batch, tuple):
            images = batch[0]
            # 根据训练阶段，batch[1] 可能是 vfm_teacher_ids (B,) 或 task_ids (B,)
            vfm_ids_from_batch = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            images = batch["image"]

            vfm_ids_from_batch = batch.get("vfm_teacher_id", None)
            # if task_training:
            #     task_ids_from_batch = batch.get("task_id", None) # 或者从 batch["labels"] 推断
            # else:
            #     task_ids_from_batch = None
        else:
            images = batch
            vfm_ids_from_batch = None

        batch_size = images.shape[0]
        B = images.shape[0]
        outputs = {}
        H, W = self.fea_size

        outputs["aux_loss"] = torch.tensor(0.0, device=images.device, dtype=images.dtype)  # 初始化辅助损失
        outputs["gate_regularization_loss"] = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        outputs["gate_entropy_loss"] = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        # --- 阶段1: VFM教师蒸馏 (每个样本 vs 每个教师) ---
        if vfm_training: # and self.vfm_projection_heads:  # 通常这种密集计算只在训练阶段进行
            if not hasattr(self, 'vfm_projection_heads') or not self.vfm_projection_heads:
                raise ValueError("vfm_training is True, but no vfm_projection_heads found or it's empty.")
            if vfm_ids_from_batch is None:
                raise ValueError("vfm_training is True, but vfm_teacher_ids are missing from batch.")
            if not (isinstance(vfm_ids_from_batch, torch.Tensor) and
                    vfm_ids_from_batch.ndim == 1 and
                    vfm_ids_from_batch.shape[0] == batch_size):
                raise ValueError(
                    "For vfm_training, ids_from_batch (e.g., batch[1] or batch['vfm_teacher_id']) "
                    "is expected to be a 1D tensor of vfm_teacher_ids with length batch_size. "
                    f"Got type: {type(vfm_ids_from_batch)}"
                    + (f", shape: {vfm_ids_from_batch.shape}" if isinstance(vfm_ids_from_batch, torch.Tensor) else "")
                )



        # vfm_student_projections = OrderedDict()  # 用于存储每个教师的投影结果
        # all_aux_losses = []
        if vfm_training:
            vfm_teacher_ids = vfm_ids_from_batch.to(images.device)

            vfm_condition_embedding = self.get_vfm_condition_embedding(vfm_teacher_ids, training=vfm_training)
            final_features_all_tokens, aux_loss, gate_list, _, gate_regularization_loss = self.backbone.forward_intermediate_features(images, vfm_condition_embedding, is_vfm_condition=True, vfm_teacher_id=vfm_teacher_ids)  # final_features_all_tokens: list, len = len(self.out_indices)

            outputs["aux_loss"] += aux_loss
            outputs["gate_regularization_loss"] += gate_regularization_loss
            # for g in gate_list:
            #     # g 的范围是 [min, max]，我们需要将其视为一个二元概率分布 [p, 1-p]
            #     # 为了计算熵，我们先将 g 视为选择 "standard_ffn" 的概率 p
            #     p = g
            #     # 熵 = - (p * log(p) + (1-p) * log(1-p))
            #     # 我们希望最大化熵，等同于最小化负熵
            #     # 添加一个小的epsilon防止log(0)
            #     epsilon = 1e-8
            #     entropy = - (p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon))
            #
            #     # 我们的损失是负熵的均值，因为优化器总是最小化损失
            #     outputs["gate_entropy_loss"] += -torch.mean(entropy)

            vfm_student_projections_output = OrderedDict()
            unique_teacher_ids_in_batch = torch.unique(vfm_teacher_ids)
            # for teacher_idx in range(self.num_vfm_teachers):
            for teacher_id_tensor in unique_teacher_ids_in_batch:
                teacher_id_str = str(teacher_id_tensor.item())  # "0", "1", ...
                mask = (vfm_teacher_ids == teacher_id_tensor)  # (B_masked,)
                # masked_features_all_tokens: (B_masked, NumTotalTokens, D_embed)
                masked_features_all_tokens = []
                for l_ind in range(len(self.out_indices)):
                    masked_features_all_tokens.append(final_features_all_tokens[l_ind][mask])

                if masked_features_all_tokens[0].shape[0] == 0:  # 如果某个教师ID没有样本
                    continue

                patch_tokens_projected = []
                for l_ind in range(len(self.out_indices)):
                    projected_feature_tokens_layer = self.vfm_projection_heads[teacher_id_str+"_"+str(l_ind)](masked_features_all_tokens[l_ind])
                    num_prefix_tokens = self.backbone.num_prefix_tokens
                    patch_tokens_projected.append(projected_feature_tokens_layer[:, num_prefix_tokens:])  # 去掉prefix tokens，只保留patch tokens
                # if not hasattr(self, 'fea_size') or len(self.fea_size) != 2:
                #     raise AttributeError(
                #         "self.fea_size (H, W) for patch token reshaping is not properly defined in __init__.")
                H_feat, W_feat = self.fea_size
                # num_patch_tokens_expected = H_feat * W_feat
                # 当前批次中，属于此教师的样本数量
                current_teacher_batch_size = patch_tokens_projected[0].shape[0]

                # if patch_tokens_projected.shape[2] != num_patch_tokens_expected:
                #     raise ValueError(
                #         f"For teacher {teacher_id_str}, number of patch tokens from projection ({patch_tokens_projected.shape[2]}) "
                #         f"does not match expected H*W ({num_patch_tokens_expected}).")
                reshaped_feature_map = []
                for l_ind in range(len(self.out_indices)):
                    reshaped_feature_map.append(patch_tokens_projected[l_ind].reshape(current_teacher_batch_size, H_feat, W_feat, -1).permute(0, 3, 1, 2).contiguous())
                # reshaped_feature_map 形状: (B_masked, D_teacher_embed, H_feat, W_feat)

                vfm_student_projections_output[teacher_id_str] = reshaped_feature_map  # 保持列表结构

            outputs["vfm_student_projections"] = vfm_student_projections_output

        if task_training:

            feature_for_tasks = OrderedDict()
            for i in range(len(self.tasks)):
                task_ids = torch.tensor([i], device=images.device, dtype=torch.long)  # 模拟 task_ids
                task_ids = task_ids.expand(batch_size)  # 扩展到 batch_size
                task_ids_embedding = self.get_task_condition_embedding(task_ids=task_ids)
                final_features_all_tokens, total_aux_loss_task, _, _, gate_regularization_loss  = self.backbone.forward_intermediate_features(images, task_ids_embedding, task_id=task_ids)

                # ########### 测试阶段一的task能力 ###########
                # vfm_teacher_ids = vfm_ids_from_batch.to(images.device)
                # vfm_condition_embedding = self.get_vfm_condition_embedding(vfm_teacher_ids, training=False)
                # final_features_all_tokens, total_aux_loss_task = self.backbone.forward_intermediate_features(images,  vfm_condition_embedding, is_vfm_condition=False)  # final_features_all_tokens: list, len = len(self.out_indices)

                #############################################################################
                for j, feat in enumerate(final_features_all_tokens):
                    patch_tokens = final_features_all_tokens[j][:, self.backbone.num_prefix_tokens:]
                    current_teacher_batch_size = patch_tokens.shape[0]
                    H_feat, W_feat = self.fea_size
                    final_features_all_tokens[j] = patch_tokens.reshape(current_teacher_batch_size, H_feat, W_feat,
                                                                        -1).permute(0, 3, 1, 2).contiguous()

                # # ########### 测试标准vit的task能力 ###########
                # final_features_all_tokens = self.backbone.forward(images, indices=self.out_indices,intermediates_only=True)  # final_features_all_tokens: list, len = len(self.out_indices)
                # total_aux_loss_task = torch.tensor(0.0, device=images.device, dtype=images.dtype)


                feature_for_tasks[self.tasks[i]] = final_features_all_tokens
                outputs["aux_loss"] += total_aux_loss_task / len(self.tasks)  # 平均每个任务的辅助损失
                outputs["gate_regularization_loss"] += gate_regularization_loss
            outputs["feature_for_tasks"] = feature_for_tasks

            # if self.backbone.num_prefix_tokens > 0:
            #     feature_for_tasks = final_features_all_tokens[:, 0]
            # else:
            #     patch_tokens_for_heads = final_features_all_tokens[:, self.backbone.num_prefix_tokens:]
            #     feature_for_tasks = torch.mean(patch_tokens_for_heads, dim=1)
            #
            # outputs["task_outputs"] = {}
            # unique_task_ids_in_batch = torch.unique(task_ids)
            # for task_idx_tensor in unique_task_ids_in_batch:
            #     task_idx_str = str(task_idx_tensor.item())
            #     if task_idx_str in self.task_heads:
            #         mask = (task_ids == task_idx_tensor)
            #         outputs["task_outputs"][task_idx_str] = self.task_heads[task_idx_str](feature_for_tasks[mask])

            # --- 默认/推理行为 ---
        elif not vfm_training and not task_training:
            global_print("Default/Inference behavior: No specific task or VFM training active.")
            vfm_condition_embedding = self.get_vfm_condition_embedding(vfm_teacher_ids, training=vfm_training)
            final_features_all_tokens, aux_loss, gate_list, topk_indices_list = self.backbone.forward_intermediate_features(images, vfm_condition_embedding, is_vfm_condition=False)  # final_features_all_tokens: list, len = len(self.out_indices)


            # final_features_all_tokens, total_aux_loss_default = self.backbone.forward_intermediate_features(images, default_condition_embedding)
            outputs["aux_loss"] = aux_loss


            if self.backbone.num_prefix_tokens > 0:
                feature_for_output = final_features_all_tokens[:, 0]
            else:
                patch_tokens_output = final_features_all_tokens[:, self.backbone.num_prefix_tokens:]
                feature_for_output = torch.mean(patch_tokens_output, dim=1)
            outputs["features"] = feature_for_output

        return outputs

    @torch.no_grad()
    def run_analysis_forward(self, image_batch, task_name):
        """
        一个干净的、用于分析的前向传播函数。

        Args:
            image_batch (torch.Tensor): 一批输入图像 (B, C, H, W)。
            task_name (str): 要分析的单个任务的名称 (e.g., 'semseg')。

        Returns:
            tuple: (final_features, gate_lists, topk_indices_lists)
                   gate_lists 和 topk_indices_lists 是包含每个MoE层输出的列表。
        """
        self.eval()  # 确保模型处于评估模式
        images = image_batch.to(next(self.parameters()).device)
        batch_size = images.shape[0]

        # 1. 根据任务名创建条件嵌入 (复用你已有的逻辑)
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found in model tasks: {self.tasks}")
        task_idx = self.tasks.index(task_name)
        task_ids = torch.tensor([task_idx] * batch_size, device=images.device, dtype=torch.long)
        condition_embedding = self.get_task_condition_embedding(task_ids=task_ids)

        # 2. 直接调用骨干网络，获取所有输出
        # 假设你的骨干网络在eval模式下会返回 (features, aux_loss, gates, indices)
        # 你需要确保你的backbone.forward_intermediate_features支持这个返回格式
        final_features_all_tokens, _, gate_list, topk_indices_list = \
            self.backbone.forward_intermediate_features(images, condition_embedding, is_vfm_condition=False)

        # 3. (可选) 将特征处理成与训练时相同的格式
        processed_features = []
        H_feat, W_feat = self.fea_size
        for feat_map in final_features_all_tokens:
            patch_tokens = feat_map[:, self.backbone.num_prefix_tokens:]
            b = patch_tokens.shape[0]
            reshaped_map = patch_tokens.reshape(b, H_feat, W_feat, -1).permute(0, 3, 1, 2).contiguous()
            processed_features.append(reshaped_map)

        return processed_features, gate_list, topk_indices_list

    def generate_gate_activation_visuals(self, image_path, task_names_to_compare, patch_size=16):
        """
        实验一：为单张图片生成并排的门控激活热力图，以对比不同任务。
        """
        # 1. 加载和预处理图像
        from torchvision import transforms
        # 使用你的数据加载器中的transformations
        # 这里是一个示例
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # 假设你的模型输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        original_img = Image.open(image_path).convert("RGB")
        image_tensor = transform(original_img).unsqueeze(0)  # 添加batch维度

        # 2. 为每个任务运行分析
        analysis_results = {}
        for task_name in task_names_to_compare:
            _, gate_list, _ = self.run_analysis_forward(image_tensor, task_name)
            # 我们只关心最后一个（最高层语义）MoE层的门控决策
            analysis_results[task_name] = gate_list[-1]

        # 3. 绘制热力图
        num_tasks = len(task_names_to_compare)
        fig, axes = plt.subplots(1, num_tasks + 1, figsize=(6 * (num_tasks + 1), 6))

        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        for i, task_name in enumerate(task_names_to_compare):
            # gate_values shape: (1, N, 1)
            gate_values = analysis_results[task_name]

            # 计算patch网格尺寸
            h, w = image_tensor.shape[2] // patch_size, image_tensor.shape[3] // patch_size
            gate_map = gate_values.reshape(h, w).cpu().numpy()

            # 上采样到原图尺寸以叠加
            gate_map_resized = F.interpolate(
                torch.tensor(gate_map).unsqueeze(0).unsqueeze(0),
                size=original_img.size[::-1],  # (H, W)
                mode='bicubic',
                align_corners=False
            ).squeeze().numpy()

            ax = axes[i + 1]
            ax.imshow(original_img)
            im = ax.imshow(gate_map_resized, cmap='magma', alpha=0.7, vmin=0, vmax=1)
            ax.set_title(f"Gate for '{task_name}'\n(Yellow: FFN, Purple: MoE)")
            ax.axis('off')

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.02)
        plt.suptitle("Gate Activation Heatmap: FFN vs. MoE Path Usage", fontsize=16)
        plt.savefig("gate_activation_heatmap_comparison.png", dpi=300)

    def generate_conditional_affinity_heatmap(self, task_dataloaders, vfm_groups, gate_threshold=0.5):
        """
        实验二：为多个任务生成条件化的专家亲和度热力图。
        只统计门控值低于阈值的token的路由决策。

        Args:
            task_dataloaders (dict): {'task_name': dataloader, ...}
            vfm_groups (dict): {'VFM_name': (start_idx, end_idx), ...}
            gate_threshold (float): 用于筛选MoE路径token的阈值。
        """
        affinity_results = {}
        num_experts = self.backbone.blocks[-1].moe_ffn_layer.num_experts  # 获取专家总数

        for task_name, loader in task_dataloaders.items():
            expert_counts = torch.zeros(num_experts, dtype=torch.long)

            for batch in tqdm(loader, desc=f"Analyzing affinity for {task_name}"):
                images = batch[0]  # 假设图像在第一个位置

                # 运行分析
                _, gate_list, topk_indices_list = self.run_analysis_forward(images, task_name)

                # 以最后一个MoE层为例
                gate_values = gate_list[-1]  # Shape: (B, N, 1)
                router_indices = topk_indices_list[-1]  # Shape: (B, N, top_k)

                # --- 核心筛选逻辑 ---
                # 1. 找到需要专家意见的token
                moe_needed_mask = (gate_values < gate_threshold).squeeze(-1)  # Shape: (B, N)

                # 2. 仅提取这些token的路由决策
                # router_indices[moe_needed_mask] 会返回一个1D张量，包含了所有被选中token的top_k个专家索引
                selected_indices = router_indices[moe_needed_mask]

                if selected_indices.numel() > 0:
                    # 3. 统计这些决策
                    counts = torch.bincount(selected_indices.flatten(), minlength=num_experts)
                    expert_counts += counts.cpu()

            # 计算频率
            total_selections = expert_counts.sum()
            affinity_results[
                task_name] = expert_counts.float() / total_selections if total_selections > 0 else np.zeros(
                num_experts)

        # --- 绘图 (复用之前的绘图函数) ---
        df = pd.DataFrame.from_dict(affinity_results, orient='index').numpy()
        df_percent = pd.DataFrame(df, index=affinity_results.keys(), columns=[f'E{i}' for i in range(num_experts)])

        plt.figure(figsize=(16, len(task_dataloaders) * 0.9))
        ax = sns.heatmap(df_percent, annot=True, fmt=".2%", cmap="viridis", linewidths=.5)
        # ... (添加VFM分组分割线和标签的代码，与之前版本相同) ...
        plt.title(f"Conditional Expert Affinity (Gate < {gate_threshold})", fontsize=16)
        plt.xlabel("Experts (Grouped by VFM Specialization)")
        plt.ylabel("Downstream Tasks")
        plt.tight_layout()
        plt.show()
