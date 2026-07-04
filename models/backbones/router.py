import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt


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



class FiLMLayer(nn.Module):
    """FiLM层：根据条件对特征进行仿射变换"""

    def __init__(self, channels, condition_dim):
        # channels: 输入特征x的通道数
        # condition_dim: 条件嵌入向量的维度
        super().__init__()
        self.channels = channels
        # 线性层，用于从条件嵌入中预测gamma和beta
        self.to_gamma_beta = nn.Linear(condition_dim, channels * 2)

        nn.init.constant_(self.to_gamma_beta.weight, 0.0)
        nn.init.constant_(self.to_gamma_beta.bias, 0.0)

    def forward(self, x, condition_embedding):
        # x: 特征张量, 形状 (B*N, C) 或 (B, N, C)
        # condition_embedding: 条件嵌入, 形状 (B*N, cond_D) 或 (B, 1, cond_D) 或 (B, cond_D)

        # 确保条件嵌入可以正确广播
        if x.ndim == 3 and condition_embedding.ndim == 2:  # x:(B,N,C), cond:(B,cond_D)
            condition_embedding = condition_embedding.unsqueeze(1)  # -> (B, 1, cond_D)
        # (B*N, C) vs (B, cond_D)的情况在ConditionedRouter中处理，这里假设维度已对齐或可广播

        # 兼容性处理：如果 condition_embedding 是 None (Stage 1)，则跳过 FiLM
        if condition_embedding is None:
            return x

        gamma_beta = self.to_gamma_beta(condition_embedding)  # 预测gamma和beta
        gamma = gamma_beta[..., :self.channels]
        beta = gamma_beta[..., self.channels:]

        # if random.random() < 0.01:
        #     print(f"gamma value: {gamma.mean().item()}")
        #     print(f"beta value: {beta.mean().item()}")
        #     print(f"x mean: {x.mean().item()}")
        return (1.0 + gamma) * x + beta  # 应用FiLM: y = gamma * x + beta

class ConditionedRouter(nn.Module):
    """条件化路由器，使用FiLM层根据条件引导路由"""

    def __init__(self, input_dim, num_experts, condition_dim, noisy_gating=True):
        # dim: token特征维度
        # num_experts: 通用Expert的数量
        # condition_dim: 条件嵌入的维度
        # noisy_gating: 是否使用带噪声的门控（在训练时增加随机性）
        super().__init__()
        self.dim = input_dim
        self.num_experts = num_experts
        self.condition_dim = condition_dim
        self.noisy_gating = noisy_gating

        # FiLM层，用于对token特征进行条件化处理
        self.film_layer = FiLMLayer(input_dim, condition_dim)

        self.W_gate_stage1 = nn.Linear(input_dim, input_dim)
        # 门控网络，作用于经过FiLM条件化的特征
        # self.W_g_base = nn.Linear(2* input_dim, num_experts)  # 输出到各个Expert的logits
        self.W_g_base = nn.Linear(input_dim, num_experts)  # 输出到各个Expert的logits
        if self.noisy_gating:
            # self.W_noise = nn.Linear(2* input_dim, num_experts)  # 预测噪声的标准差
            self.W_noise = nn.Linear(input_dim, num_experts)  # 预测噪声的标准差

    def forward(self, x_tokens, condition_embedding, is_vfm_condition=False, routing_mask=None):
        # x_tokens: token特征
        # condition_embedding: VFM ID或Task ID的嵌入，或在阶段一被丢弃时为None
        # is_stage1: 一个布尔标志，用于区分当前是哪个训练阶段

        original_shape = x_tokens.shape
        if x_tokens.ndim == 3:
            x_tokens = x_tokens.view(-1, self.dim)

        # --- 1. 计算基础 Logits (模型的"直觉") ---
        # 这是所有路由决策的基础，不依赖任何外部条件。
        # base_logits = self.W_g_base(x_tokens)

        # # 如果没有提供条件，或者条件被丢弃，则直接返回基础logits
        # if condition_embedding is None:
        #     final_logits = base_logits
        #     conditioned_feature_for_noise = x_tokens  # 用于噪声预测的特征
        # else:
        # --- 2. 扩展条件嵌入 ---
        if condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
            num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
            condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)
        else:
            condition_embedding_expanded = condition_embedding

       ######################Film router ##############################
        # 1. FiLM 调制
        conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)

        # 2. Gate 计算 (全阶段统一)
        # 这个 Gate 决定了 Condition 对当前 Token 的影响程度
        gate = torch.sigmoid(self.W_gate_stage1(x_tokens))

        # 3. 特征融合 (全阶段统一)
        # 这种残差形式 (Original + Gate * Delta) 非常稳健
        final_feature_to_project = x_tokens + gate * (conditioned_x - x_tokens)
        conditioned_feature_for_noise = final_feature_to_project


        # #######   concat #######################################
        # final_feature_to_project = torch.cat([x_tokens, condition_embedding_expanded], dim=-1)
        # conditioned_feature_for_noise = final_feature_to_project


        # 4. 计算 Logits
        final_logits = self.W_g_base(final_feature_to_project)

        # # --- 3. 使用FiLM层处理特征 ---
        # # conditioned_x 是被条件"调制"过的特征，包含了条件的引导信息
        # conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)
        #
        # # --- 4. 根据不同阶段，决定如何使用调制后的特征 ---
        # if is_vfm_condition:
        #     # --- 阶段一: GatedFusion + 随机丢弃 ---
        #     # 计算调制后的logits
        #     # conditioned_logits = self.W_g_base(conditioned_x)
        #
        #     # 计算门控信号 (注意：门是基于原始的x_tokens生成的，这更符合逻辑)
        #     gate = torch.sigmoid(self.W_gate_stage1(x_tokens))
        #
        #     final_feature_to_project = x_tokens + gate * (conditioned_x - x_tokens) #(1-gate) * x_tokens + gate * conditioned_x
        #     # final_logits = self.W_g_base(final_feature_to_project)
        #     conditioned_feature_for_noise = final_feature_to_project
        #     # 融合logits: 基础logits + 门控后的(条件logits - 基础logits)
        #     # 这种形式的融合更稳定: base + gate * (cond - base)
        #     # global_print(base_logits.shape, conditioned_logits.shape, gate.shape)
        #     # final_logits = base_logits + gate * (conditioned_logits - base_logits)
        #     # conditioned_feature_for_noise = x_tokens + gate * (conditioned_x - x_tokens)
        #
        # else:  # 阶段二
        #     # --- 阶段二: 标准的条件化路由 ---
        #     # 直接使用被FiLM调制后的特征来计算logits
        #
        #     conditioned_feature_for_noise = conditioned_x
        # final_logits = self.W_g_base(conditioned_feature_for_noise)

        # ============================================================
        # Step 8: 添加 Expert Dropout (强制随机丢弃) -> 解决崩塌的关键
        # ============================================================
        # 作用：强行把某些专家的 logits 设为 -inf，Router 必须学会选"备胎"
        if self.training:
            # 1. 生成随机 Mask (20% 概率丢弃)
            # 注意：这里必须用 final_logits 来生成形状
            dropout_mask = torch.rand_like(final_logits) < 0.1

            # 2. 应用 Mask
            # 使用 masked_fill 将被选中的位置设为负无穷
            # 这样 Softmax 之后概率为 0，彻底阻断该路径
            final_logits = final_logits.masked_fill(dropout_mask, float('-inf'))

        # --- 5. 添加噪声 (可选) ---
        if self.noisy_gating and self.training:
            # 我们使用一个统一的变量来获取用于预测噪声的特征
            noise_std = F.softplus(self.W_noise(conditioned_feature_for_noise))
            noise = torch.randn_like(final_logits) * noise_std
            final_logits += noise

        if len(original_shape) == 3:
            final_logits = final_logits.view(original_shape[0], original_shape[1], self.num_experts)

        return final_logits

    # def forward(self, x_tokens, condition_embedding):
    #     # x_tokens: token特征, 形状 (B*N, D) (batch_size * num_tokens, dim)
    #     # condition_embedding: 条件嵌入, 形状 (B, cond_D) 或已扩展为 (B*N, cond_D)
    #
    #     # 如果condition_embedding是 (B, cond_D)，需要为每个token复制
    #     if x_tokens.ndim == 2 and condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
    #         # 假设 x_tokens 是 (B*N, D)，condition_embedding 是 (B, cond_D)
    #         num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
    #         condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)  # -> (B*N, cond_D)
    #     elif x_tokens.ndim == 2 and condition_embedding.ndim == 2 and x_tokens.size(0) == condition_embedding.size(0):
    #         condition_embedding_expanded = condition_embedding  # 已经是对齐的(B*N, cond_D)
    #     elif x_tokens.ndim == 3 and condition_embedding.ndim == 2:  # x_tokens (B,N,D), cond (B,cond_D)
    #         condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, x_tokens.size(1), -1)  # (B,N,cond_D)
    #         x_tokens = x_tokens.reshape(-1, self.dim)  # (B*N, D)
    #         condition_embedding_expanded = condition_embedding_expanded.reshape(-1, self.condition_dim)  # (B*N, cond_D)
    #     else:
    #         raise ValueError(
    #             f"Shape mismatch or unhandled case for x_tokens ({x_tokens.shape}) and condition_embedding ({condition_embedding.shape})")
    #
    #     conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)  # FiLM调制
    #
    #     router_logits = self.W_g(conditioned_x)  # 计算路由logits
    #     if self.noisy_gating and self.training:  # 训练时且启用noisy_gating
    #         noise_std = F.softplus(self.W_noise(conditioned_x))  # softplus确保标准差为正
    #         noise = torch.randn_like(router_logits) * noise_std  # 采样噪声
    #         router_logits += noise  # 添加噪声
    #
    #     return F.softmax(router_logits, dim=-1)  # 通过softmax得到路由权重


class AdvancedConditionedRouter(nn.Module):
    def __init__(self, input_dim, num_experts, condition_dim, noisy_gating=True, hidden_dim_ratio=0.5):
        super().__init__()
        self.dim = input_dim
        self.num_experts = num_experts
        self.condition_dim = condition_dim
        self.noisy_gating = noisy_gating

        # 【升级1】独立的 LayerNorm
        # 这是 DeepSeek/Mixtral 稳定训练的关键，解耦模长与方向
        self.router_norm = nn.LayerNorm(input_dim)

        # FiLM 层
        self.film_layer = FiLMLayer(input_dim, condition_dim)

        # Stage 1 的门控网络
        self.W_gate_stage1 = nn.Linear(input_dim, input_dim)

        # 【升级2】MLP Router (替代原来的 W_g_base)
        # 给 Router 一个"脑子"，让它能处理复杂的条件逻辑
        hidden_dim = int(input_dim * hidden_dim_ratio)
        self.router_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh 比 ReLU/GELU 在 Router 中更稳定
            nn.Linear(hidden_dim, num_experts)
        )

        if self.noisy_gating:
            self.W_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x_tokens, condition_embedding, is_vfm_condition=False, routing_mask=None):
        original_shape = x_tokens.shape
        if x_tokens.ndim == 3:
            x_tokens = x_tokens.view(-1, self.dim)

        # ============================================================
        # Step 1: 归一化 (至关重要)
        # ============================================================
        # 无论后续怎么处理，Router 看到的特征都应该是标准化的
        x_norm = self.router_norm(x_tokens)

        # ============================================================
        # Step 2: 扩展条件嵌入
        # ============================================================
        if condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
            num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
            condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)
        else:
            condition_embedding_expanded = condition_embedding

        # ============================================================
        # Step 3: FiLM 调制 (基于归一化后的特征)
        # ============================================================
        # conditioned_x: 携带了 Task/VFM 意图的特征
        conditioned_x = self.film_layer(x_norm, condition_embedding_expanded)

        # ============================================================
        # Step 4: 阶段性逻辑处理
        # ============================================================
        if is_vfm_condition:
            # --- 阶段一: 软性门控融合 ---
            # 计算门控 (基于归一化特征，防止 Sigmoid 饱和)
            # gate: (B*N, D)
            gate = torch.sigmoid(self.W_gate_stage1(x_norm))

            # 融合: 原始特征 + 门控 * (调制带来的增量)
            # 这种 Residual 形式保证了如果条件没用，Router 至少能退化回只看 x_norm
            final_feature = x_norm + gate * (conditioned_x - x_norm)
        else:
            # --- 阶段二: 强引导 ---
            # 直接使用调制后的特征，强制 Router 关注 Task ID
            final_feature = conditioned_x

        # ============================================================
        # Step 5: 计算 Logits (MLP)
        # ============================================================
        final_logits = self.router_mlp(final_feature)

        # ==========================================================
        # Step 6: 应用 Mask (如果有)
        # ==========================================================
        if routing_mask is not None:
            if routing_mask.ndim == 2 and routing_mask.shape[0] != final_logits.shape[0]:
                num_tokens_per_sample = final_logits.shape[0] // routing_mask.shape[0]
                routing_mask = routing_mask.repeat_interleave(num_tokens_per_sample, dim=0)
            final_logits += routing_mask

        # ============================================================
        # Step 7: 添加噪声
        # ============================================================
        if self.noisy_gating and self.training:
            # 噪声也基于 final_feature 生成，保证相关性
            noise_std = F.softplus(self.W_noise(final_feature))
            noise = torch.randn_like(final_logits) * noise_std
            final_logits += noise

        # # ============================================================
        # # Step 8: 添加 Expert Dropout (强制随机丢弃) -> 解决崩塌的关键
        # # ============================================================
        # # 作用：强行把某些专家的 logits 设为 -inf，Router 必须学会选"备胎"
        # if self.training:
        #     # 1. 生成随机 Mask (20% 概率丢弃)
        #     # 注意：这里必须用 final_logits 来生成形状
        #     dropout_mask = torch.rand_like(final_logits) < 0.2
        #
        #     # 2. 应用 Mask
        #     # 使用 masked_fill 将被选中的位置设为负无穷
        #     # 这样 Softmax 之后概率为 0，彻底阻断该路径
        #     final_logits = final_logits.masked_fill(dropout_mask, float('-inf'))

        # ============================================================
        # Step 9: 恢复形状
        # ============================================================
        # 建议把 Dropout 放在 view 之前，计算上通常更方便，效果是一样的
        if len(original_shape) == 3:
            final_logits = final_logits.view(original_shape[0], original_shape[1], self.num_experts)

        return final_logits



# global_print("Using no structure distill!")

################################ structure distill#########################

# class StructureConditionedRouter(nn.Module):
#     """条件化路由器，使用FiLM层根据条件引导路由"""
#
#     def __init__(self, input_dim, num_experts, condition_dim, noisy_gating=True):
#         # dim: token特征维度
#         # num_experts: 通用Expert的数量
#         # condition_dim: 条件嵌入的维度
#         # noisy_gating: 是否使用带噪声的门控（在训练时增加随机性）
#         super().__init__()
#         self.dim = input_dim
#         self.num_experts = num_experts
#         self.condition_dim = condition_dim
#         self.noisy_gating = noisy_gating
#
#         # FiLM层，用于对token特征进行条件化处理
#         self.film_layer = FiLMLayer(input_dim, condition_dim)
#
#         self.W_gate_stage1 = nn.Linear(input_dim, input_dim)
#         # 门控网络，作用于经过FiLM条件化的特征
#         self.W_g_base = nn.Linear(input_dim, num_experts)  # 输出到各个Expert的logits
#         if self.noisy_gating:
#             self.W_noise = nn.Linear(input_dim, num_experts)  # 预测噪声的标准差
#
#     def forward(self, x_tokens, condition_embedding, is_vfm_condition=False, routing_mask=None):
#         # x_tokens: token特征
#         # condition_embedding: VFM ID或Task ID的嵌入，或在阶段一被丢弃时为None
#         # is_stage1: 一个布尔标志，用于区分当前是哪个训练阶段
#
#         original_shape = x_tokens.shape
#         if x_tokens.ndim == 3:
#             x_tokens = x_tokens.view(-1, self.dim)
#
#         # --- 1. 计算基础 Logits (模型的"直觉") ---
#         # 这是所有路由决策的基础，不依赖任何外部条件。
#         # base_logits = self.W_g_base(x_tokens)
#
#         # # 如果没有提供条件，或者条件被丢弃，则直接返回基础logits
#         # if condition_embedding is None:
#         #     final_logits = base_logits
#         #     conditioned_feature_for_noise = x_tokens  # 用于噪声预测的特征
#         # else:
#         # --- 2. 扩展条件嵌入 ---
#         if condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
#             num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
#             condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)
#         else:
#             condition_embedding_expanded = condition_embedding
#
#         # --- 3. 使用FiLM层处理特征 ---
#         # conditioned_x 是被条件"调制"过的特征，包含了条件的引导信息
#         conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)
#         conditioned_feature_for_noise = conditioned_x
#
#         # # --- 4. 根据不同阶段，决定如何使用调制后的特征 ---
#         # if is_vfm_condition:
#         #     # --- 阶段一: GatedFusion + 随机丢弃 ---
#         #     # 计算调制后的logits
#         #     # conditioned_logits = self.W_g_base(conditioned_x)
#         #
#         #     # 计算门控信号 (注意：门是基于原始的x_tokens生成的，这更符合逻辑)
#         #     gate = torch.sigmoid(self.W_gate_stage1(x_tokens))
#         #
#         #     final_feature_to_project = x_tokens + gate * (conditioned_x - x_tokens) #(1-gate) * x_tokens + gate * conditioned_x
#         #     # final_logits = self.W_g_base(final_feature_to_project)
#         #     conditioned_feature_for_noise = final_feature_to_project
#         #     # 融合logits: 基础logits + 门控后的(条件logits - 基础logits)
#         #     # 这种形式的融合更稳定: base + gate * (cond - base)
#         #     # global_print(base_logits.shape, conditioned_logits.shape, gate.shape)
#         #     # final_logits = base_logits + gate * (conditioned_logits - base_logits)
#         #     # conditioned_feature_for_noise = x_tokens + gate * (conditioned_x - x_tokens)
#         #
#         # else:  # 阶段二
#         #     # --- 阶段二: 标准的条件化路由 ---
#         #     # 直接使用被FiLM调制后的特征来计算logits
#         #
#         #     conditioned_feature_for_noise = conditioned_x
#
#         final_logits = self.W_g_base(conditioned_feature_for_noise)
#
#         # ==========================================================
#         # ===               【核心修改】应用路由Mask               ===
#         # ==========================================================
#         if routing_mask is not None:
#             # 确保mask的形状可以广播到logits上
#             if routing_mask.ndim == 2 and routing_mask.shape[0] != final_logits.shape[0]:
#                 # 假设 mask 是 (B, num_experts), logits 是 (B*N, num_experts)
#                 num_tokens_per_sample = final_logits.shape[0] // routing_mask.shape[0]
#                 routing_mask = routing_mask.repeat_interleave(num_tokens_per_sample, dim=0)
#
#             # 将mask加到logits上。值为-inf的位置会使softmax后的概率趋近于0
#             final_logits += routing_mask
#         # ==========================================================
#
#
#
#         # --- 5. 添加噪声 (可选) ---
#         if self.noisy_gating and self.training:
#             # 我们使用一个统一的变量来获取用于预测噪声的特征
#             noise_std = F.softplus(self.W_noise(conditioned_feature_for_noise))
#             noise = torch.randn_like(final_logits) * noise_std
#             final_logits += noise
#
#         if len(original_shape) == 3:
#             final_logits = final_logits.view(original_shape[0], original_shape[1], self.num_experts)
#
#         return final_logits
#
#     # def forward(self, x_tokens, condition_embedding):
#     #     # x_tokens: token特征, 形状 (B*N, D) (batch_size * num_tokens, dim)
#     #     # condition_embedding: 条件嵌入, 形状 (B, cond_D) 或已扩展为 (B*N, cond_D)
#     #
#     #     # 如果condition_embedding是 (B, cond_D)，需要为每个token复制
#     #     if x_tokens.ndim == 2 and condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
#     #         # 假设 x_tokens 是 (B*N, D)，condition_embedding 是 (B, cond_D)
#     #         num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
#     #         condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)  # -> (B*N, cond_D)
#     #     elif x_tokens.ndim == 2 and condition_embedding.ndim == 2 and x_tokens.size(0) == condition_embedding.size(0):
#     #         condition_embedding_expanded = condition_embedding  # 已经是对齐的(B*N, cond_D)
#     #     elif x_tokens.ndim == 3 and condition_embedding.ndim == 2:  # x_tokens (B,N,D), cond (B,cond_D)
#     #         condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, x_tokens.size(1), -1)  # (B,N,cond_D)
#     #         x_tokens = x_tokens.reshape(-1, self.dim)  # (B*N, D)
#     #         condition_embedding_expanded = condition_embedding_expanded.reshape(-1, self.condition_dim)  # (B*N, cond_D)
#     #     else:
#     #         raise ValueError(
#     #             f"Shape mismatch or unhandled case for x_tokens ({x_tokens.shape}) and condition_embedding ({condition_embedding.shape})")
#     #
#     #     conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)  # FiLM调制
#     #
#     #     router_logits = self.W_g(conditioned_x)  # 计算路由logits
#     #     if self.noisy_gating and self.training:  # 训练时且启用noisy_gating
#     #         noise_std = F.softplus(self.W_noise(conditioned_x))  # softplus确保标准差为正
#     #         noise = torch.randn_like(router_logits) * noise_std  # 采样噪声
#     #         router_logits += noise  # 添加噪声
#     #
#     #     return F.softmax(router_logits, dim=-1)  # 通过softmax得到路由权重

class StructureConditionedRouter(nn.Module):
    def __init__(self, input_dim, num_experts, condition_dim, noisy_gating=True, top_k=1):
        super().__init__()
        self.dim = input_dim
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.top_k = top_k
        # FiLM层
        self.film_layer = FiLMLayer(input_dim, condition_dim)

        # 核心门控层
        self.W_g_base = nn.Linear(input_dim, num_experts)
        if self.noisy_gating:
            self.W_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x_tokens, condition_embedding, is_vfm_condition=False, routing_mask=None, override_top_k=None):
        """
        top_k: 可以在调用时动态指定。
               Phase 1 设为 1 或 2 (组内竞争)
               Phase 2 设为 3 或 4 (全局组合)
        """
        original_shape = x_tokens.shape
        if x_tokens.ndim == 3:
            x_tokens = x_tokens.view(-1, self.dim)

        # --- 1. 条件扩展 & FiLM ---
        if condition_embedding.ndim == 2 and x_tokens.size(0) != condition_embedding.size(0):
            num_tokens_per_sample = x_tokens.size(0) // condition_embedding.size(0)
            condition_embedding_expanded = condition_embedding.repeat_interleave(num_tokens_per_sample, dim=0)
        else:
            condition_embedding_expanded = condition_embedding

        # FiLM 调制：这一步至关重要，让特征带上 Task/VFM 的印记
        conditioned_x = self.film_layer(x_tokens, condition_embedding_expanded)

        # --- 2. 计算 Logits ---
        logits = self.W_g_base(conditioned_x)

        # --- 3. 噪声机制 (Standard Gating) ---
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.W_noise(conditioned_x))
            noise = torch.randn_like(logits) * noise_std
            logits += noise

        # --- 4. 应用 Mask (Phase 1 关键步骤) ---
        # 这里的 mask 应该是 [B*N, num_experts] 或者 [B, num_experts]
        # 值应该为 0 (允许) 或 -inf (禁止)
        if routing_mask is not None:
            if routing_mask.ndim == 2 and routing_mask.shape[0] != logits.shape[0]:
                num_tokens_per_sample = logits.shape[0] // routing_mask.shape[0]
                routing_mask = routing_mask.repeat_interleave(num_tokens_per_sample, dim=0)

            # 加上 mask (-inf 会让 softmax 归零)
            logits = logits + routing_mask

        # --- 5. 计算路由概率 (用于 Aux Loss) ---
        # 注意：必须在 TopK 之前算 Softmax，这样才能反映真实的门控分布
        routing_probs = F.softmax(logits, dim=-1)

        # --- 6. Top-K 选择 ---
        # --- 6. 动态 Top-K 选择 ---
        # 优先使用传入的 override_top_k，否则使用实例默认值
        top_k = override_top_k if override_top_k is not None else self.top_k

        # 这里的 top_k 是动态传入的，实现了 Phase 1 和 Phase 2 的灵活切换
        top_k_probs, top_k_indices = torch.topk(routing_probs, top_k, dim=-1)

        # 归一化 Top-K 权重 (使其和为1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)

        if len(original_shape) == 3:
            # 恢复形状返回
            B, N = original_shape[0], original_shape[1]
            return top_k_probs.view(B, N, top_k), top_k_indices.view(B, N, top_k), routing_probs.view(B, N, self.num_experts)

        return top_k_probs, top_k_indices, routing_probs


class TaskBiasGenerator(nn.Module):
    def __init__(self, num_groups, num_tasks=10):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, num_groups)
        nn.init.zeros_(self.task_embed.weight)

    def forward(self, task_id, batch_size):
        # 如果没有 task_id，返回 None，我们在 Router 里利用广播机制处理默认值
        if task_id is None:
            return None

        if isinstance(task_id, int):
            task_id = torch.tensor([task_id], device=self.task_embed.weight.device)

        bias = self.task_embed(task_id)  # (1, num_groups)
        weights = torch.sigmoid(bias)
        # 返回 (1, num_groups) 或 (B, num_groups)，取决于输入
        # 如果所有 sample 同一个 task，保持 (1, num_groups) 以利用广播
        if weights.shape[0] == 1:
            return weights
        else:
            return weights.expand(batch_size, -1)


class DecoupledGroupRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_experts_per_group, condition_dim,
                 noisy_gating=True, num_tasks=0):
        super().__init__()
        self.dim = input_dim
        self.num_experts = num_experts
        self.num_experts_per_group = num_experts_per_group
        self.num_groups = num_experts // num_experts_per_group
        self.noisy_gating = noisy_gating
        self.num_tasks = num_tasks

        self.router_linear = nn.Linear(input_dim, num_experts)

        if self.noisy_gating:
            self.noise_linear = nn.Linear(input_dim, num_experts)

        if self.num_tasks == 0:
            self.task_bias_gen = None
        else:
            self.task_bias_gen = TaskBiasGenerator(self.num_groups, num_tasks)

        self.film_layer = FiLMLayer(input_dim, condition_dim)

    def forward(self, x, condition_embedding=None, task_id=None, override_group_topk=None):
        """
        x: (B, N, D) [推荐] 或 (B*N, D)
        """
        k = override_group_topk if override_group_topk is not None else 1

        is_flattened = (x.ndim == 2)
        if is_flattened:
            B_temp = 1  # 无法准确获取，假设为1，注意这在Stage 2可能影响Task Bias的Expand
            x_flat = x
        else:
            B, N, D = x.shape
            x_flat = x.view(-1, D)

        # 2. FiLM (Optional)
        # x_routed = self.film_layer(x_flat, condition_embedding)
        x_routed = x_flat

        # 3. 计算 Logits
        logits = self.router_linear(x_routed)

        # 4. Noisy Gating
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.noise_linear(x_routed))
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        # 5. 变形为分组结构: (B*N, G, K_total)
        logits_grouped = logits.view(-1, self.num_groups, self.num_experts_per_group)

        # 6. 组内 Softmax
        probs_grouped = F.softmax(logits_grouped, dim=-1)  # (B*N, G, K_total)

        # 7. 组内 Top-K
        topk_vals, topk_indices = torch.topk(probs_grouped, k=k, dim=-1)

        # 归一化 (针对 K > 1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-6)
        # Shape: (B*N, G, K)

        # 8. 计算全局索引
        group_offsets = torch.arange(self.num_groups, device=x.device) * self.num_experts_per_group
        group_offsets = group_offsets.view(1, self.num_groups, 1)

        final_indices = topk_indices + group_offsets  # (B*N, G, K)

        # 9. 应用 Task Bias (Stage 2)
        group_scores = topk_vals

        if self.task_bias_gen is not None and task_id is not None:
            if not is_flattened:
                # 恢复 (B, N, G, K)
                group_scores = group_scores.view(B, N, self.num_groups, k)

                # 获取 Task Weights: (B, G)
                task_weights = self.task_bias_gen(task_id, B)

                if task_weights is not None:
                    # 扩展维度: (B, 1, G, 1) 以匹配 (B, N, G, K)
                    w_expanded = task_weights.unsqueeze(1).unsqueeze(-1)
                    group_scores = group_scores * w_expanded

                # 变回 Flatten: (B*N, G, K)
                group_scores = group_scores.view(-1, self.num_groups, k)
            else:
                # Flatten情况兼容
                task_weights = self.task_bias_gen(task_id, 1)
                w_expanded = task_weights.unsqueeze(-1)
                group_scores = group_scores * w_expanded

        # 10. 最终展平
        # 返回形状: (B*N, G*K)
        final_weights_flat = group_scores.view(-1, self.num_groups * k)
        final_indices_flat = final_indices.view(-1, self.num_groups * k)

        return final_weights_flat, final_indices_flat, probs_grouped
