import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.backbones.router import ConditionedRouter, AdvancedConditionedRouter


class UpcycledMoHAttention(nn.Module):
    """
    支持 Upcycling (Head Splitting) 和 Shared Heads 的 MoH Attention
    """

    def __init__(self, base_attn, dim, num_heads, num_selected_heads,
                 expand_ratio=2,  # 扩展倍数，例如 2 表示头数翻倍
                 num_shared_heads=0,  # 共享头数量 (始终激活)
                 noisy_gating=True,
                 noise_std=0.02):  # 初始化的噪声强度
        super().__init__()

        # 1. 计算新的头数配置
        self.orig_num_heads = num_heads
        self.total_num_heads = int(num_heads * expand_ratio)  # 扩展后的总头数
        self.num_shared_heads = num_shared_heads

        # 路由头数量 = 总头数 - 共享头数
        self.num_routed_heads = self.total_num_heads - num_shared_heads
        self.num_selected_heads = num_selected_heads  # Top-K (针对 Routed Heads)

        # 确保配置合理
        assert self.num_routed_heads > 0, "总头数必须大于共享头数"
        assert self.num_selected_heads <= self.num_routed_heads, "TopK 不能超过路由头总数"

        self.head_dim = dim // num_heads  # Head Dim 保持不变
        self.scale = self.head_dim ** -0.5
        self.noisy_gating = noisy_gating

        # 2. 定义新的更大的线性层 (从 base_attn 扩展而来)
        # QKV: 输出维度变大 (3 * total_heads * head_dim)
        new_qkv_dim = 3 * self.total_num_heads * self.head_dim
        self.qkv = nn.Linear(dim, new_qkv_dim, bias=base_attn.qkv.bias is not None)

        # Proj: 输入维度变大 (total_heads * head_dim)
        new_proj_in_dim = self.total_num_heads * self.head_dim
        self.proj = nn.Linear(new_proj_in_dim, dim, bias=base_attn.proj.bias is not None)

        self.attn_drop = base_attn.attn_drop
        self.proj_drop = base_attn.proj_drop

        # 3. Router (只负责路由那些 非Shared 的头)
        self.router = nn.Linear(dim, self.num_routed_heads)
        if noisy_gating:
            self.w_noise = nn.Linear(dim, self.num_routed_heads)

        # 4. 执行 Upcycling 初始化 (核心逻辑)
        self._upcycle_weights(base_attn, expand_ratio, noise_std)

    def _upcycle_weights(self, base_attn, expand_ratio, noise_std):
        """
        核心函数：复制权重并添加噪声
        """
        with torch.no_grad():
            # --- 处理 QKV 权重 ---
            # 原始 shape: [3 * H_old * D, Dim]
            old_qkv_w = base_attn.qkv.weight
            dim = old_qkv_w.shape[1]

            # Reshape 为 [3, H_old, D, Dim] 以便在 Head 维度操作
            old_qkv_w = old_qkv_w.view(3, self.orig_num_heads, self.head_dim, dim)

            # 在 Head 维度 (dim=1) 进行复制
            # repeat_interleave 会把 [H1, H2] 变成 [H1, H1, H2, H2] (聚类效果更好)
            # 或者 repeat 变成 [H1, H2, H1, H2]。这里推荐 repeat_interleave
            new_qkv_w = old_qkv_w.repeat_interleave(int(expand_ratio), dim=1)

            # 添加噪声 (Symmetry Breaking)
            # 只给复制出来的部分加噪声？或者全加？
            # 建议：全加微小噪声，或者保持第一份副本不变，给后面的副本加噪声。
            # 为了简单有效，我们给整体加一个极小的扰动，让 Router 自己区分。
            noise = torch.randn_like(new_qkv_w) * noise_std
            new_qkv_w = new_qkv_w + noise

            # Reshape 回 [3 * H_new * D, Dim] 并赋值
            self.qkv.weight.copy_(new_qkv_w.reshape(-1, dim))

            # 处理 QKV Bias
            if base_attn.qkv.bias is not None:
                old_qkv_b = base_attn.qkv.bias.view(3, self.orig_num_heads, self.head_dim)
                new_qkv_b = old_qkv_b.repeat_interleave(int(expand_ratio), dim=1)
                self.qkv.bias.copy_(new_qkv_b.reshape(-1))

            # --- 处理 Projection 权重 ---
            # 原始 shape: [Dim, H_old * D] -> 这里的 H 在 dim=1
            old_proj_w = base_attn.proj.weight

            # Reshape 为 [Dim, H_old, D]
            old_proj_w = old_proj_w.view(dim, self.orig_num_heads, self.head_dim)

            # 复制
            new_proj_w = old_proj_w.repeat_interleave(int(expand_ratio), dim=1)

            # 【关键】缩放 (Scaling)
            # 因为头数翻倍了，如果直接相加，输出值的方差会变大。
            # 建议将权重除以 expand_ratio，或者依靠 LayerNorm 调整。
            # 这里我们保守一点，加上噪声，同时除以 sqrt(expand_ratio) 或者保持原样依靠 Residual。
            # 通常加噪声即可，Router 会自动学会关掉不用的头。
            new_proj_w = new_proj_w + (torch.randn_like(new_proj_w) * noise_std)

            # 稍微缩小权重以抵消头数增加带来的数值膨胀 (可选，视 expand_ratio 大小而定)
            # new_proj_w = new_proj_w / (expand_ratio ** 0.5)

            self.proj.weight.copy_(new_proj_w.reshape(dim, -1))

            if base_attn.proj.bias is not None:
                self.proj.bias.copy_(base_attn.proj.bias)

            print(f"Upcycling done: Heads {self.orig_num_heads} -> {self.total_num_heads} "
                  f"(Shared: {self.num_shared_heads}, Routed: {self.num_routed_heads})")

    def forward(self, x):
        B, N, C = x.shape

        # 1. 计算所有 Head 的 QKV (包括 Shared 和 Routed)
        # qkv: (B, N, 3 * H_total * D)
        qkv = self.qkv(x)
        # reshape: (B, N, 3, H_total, D) -> permute -> (3, B, H_total, N, D)
        qkv = qkv.reshape(B, N, 3, self.total_num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H_total, N, D)

        # 2. 计算 Attention Score (所有 Head 一起算，利用矩阵乘法并行优势)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v)  # (B, H_total, N, D)

        # 3. 拆分 Shared 和 Routed
        # 假设前 num_shared_heads 个是共享的 (你可以自由设定顺序，这里假设 shared 在前)
        if self.num_shared_heads > 0:
            x_shared = x_attn[:, :self.num_shared_heads, :, :]  # (B, H_shared, N, D)
            x_routed = x_attn[:, self.num_shared_heads:, :, :]  # (B, H_routed, N, D)
        else:
            x_shared = None
            x_routed = x_attn

        # 4. 路由逻辑 (只针对 x_routed)
        router_logits = self.router(x)  # (B, N, H_routed)

        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.w_noise(x))
            noise = torch.randn_like(router_logits) * noise_std
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 选择
        top_k_weights, selected_indices = torch.topk(router_logits, self.num_selected_heads, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 构建 Mask 并应用
        # mask: (B, N, H_routed)
        mask = F.one_hot(selected_indices, num_classes=self.num_routed_heads).float().sum(dim=2)
        # 扩展 mask: (B, H_routed, N, 1)
        mask_expanded = mask.permute(0, 2, 1).unsqueeze(-1)

        # 应用 Mask 到 Routed Heads
        x_routed_masked = x_routed * mask_expanded

        # 5. 拼接回 (Shared + Routed_Masked)
        if x_shared is not None:
            # 拼接在 Head 维度 (dim=1)
            x_final = torch.cat([x_shared, x_routed_masked], dim=1)
        else:
            x_final = x_routed_masked

        # 6. 最终投影
        # x_final: (B, H_total, N, D) -> (B, N, H_total * D)
        x_final = x_final.transpose(1, 2).reshape(B, N, -1)
        x_final = self.proj(x_final)
        x_final = self.proj_drop(x_final)

        # 7. Aux Loss (只基于 Routed 部分)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            importance = router_probs.sum(dim=(0, 1))
            load = mask.sum(dim=(0, 1))
            aux_loss = torch.sum(importance * load) / (B * N)

        return x_final, aux_loss, router_probs, selected_indices


class MoHAttention(nn.Module):
    def __init__(self, base_attn, dim, num_heads, num_selected_heads,
                 condition_dim,  # 必须传入
                 expand_ratio=2,  # <--- 新增：需传入
                 num_shared_heads=0,  # <--- 新增：需传入
                 noise_std=0.02,  # <--- 新增：需传入
                 noisy_gating=True):
        super().__init__()

        # 1. 计算头数配置
        self.orig_num_heads = num_heads
        self.total_num_heads = int(num_heads * expand_ratio)  # 扩展后的总头数
        self.num_shared_heads = num_shared_heads

        # 路由头数量 = 总头数 - 共享头数
        self.num_routed_heads = self.total_num_heads - num_shared_heads
        self.num_selected_heads = num_selected_heads  # Top-K (针对 Routed Heads)

        # 确保配置合理
        assert self.num_routed_heads > 0, "总头数必须大于共享头数"
        assert self.num_selected_heads <= self.num_routed_heads, "TopK 不能超过路由头总数"

        self.head_dim = dim // num_heads  # Head Dim 保持不变
        self.scale = self.head_dim ** -0.5
        self.noisy_gating = noisy_gating

        # 2. 定义新的更大的线性层 (从 base_attn 扩展而来)
        # QKV: 输出维度变大 (3 * total_heads * head_dim)
        new_qkv_dim = 3 * self.total_num_heads * self.head_dim
        self.qkv = nn.Linear(dim, new_qkv_dim, bias=base_attn.qkv.bias is not None)

        # Proj: 输入维度变大 (total_heads * head_dim)
        new_proj_in_dim = self.total_num_heads * self.head_dim
        self.proj = nn.Linear(new_proj_in_dim, dim, bias=base_attn.proj.bias is not None)

        self.attn_drop = base_attn.attn_drop
        self.proj_drop = base_attn.proj_drop

        # ============================================================
        # 【核心修改】替换为高级路由器
        # ============================================================
        # 注意：Router 只需要为 [Routed Heads] 打分，不需要管 Shared Heads
        self.router = AdvancedConditionedRouter(
            input_dim=dim,
            num_experts=self.num_routed_heads,  # <--- 修正：只路由非共享的头
            condition_dim=condition_dim,
            noisy_gating=noisy_gating,
            hidden_dim_ratio=0.5
        )

        # 4. 执行 Upcycling 初始化
        self._upcycle_weights(base_attn, expand_ratio, noise_std)

    def _upcycle_weights(self, base_attn, expand_ratio, noise_std):
        """
        核心函数：复制权重并添加噪声
        """
        with torch.no_grad():
            # --- 处理 QKV 权重 ---
            # 原始 shape: [3 * H_old * D, Dim]
            old_qkv_w = base_attn.qkv.weight
            dim = old_qkv_w.shape[1]

            # Reshape 为 [3, H_old, D, Dim]
            old_qkv_w = old_qkv_w.view(3, self.orig_num_heads, self.head_dim, dim)

            # 在 Head 维度 (dim=1) 进行复制
            new_qkv_w = old_qkv_w.repeat_interleave(int(expand_ratio), dim=1)

            # 添加噪声 (Symmetry Breaking)
            noise = torch.randn_like(new_qkv_w) * noise_std
            new_qkv_w = new_qkv_w + noise

            # 赋值
            self.qkv.weight.copy_(new_qkv_w.reshape(-1, dim))

            # 处理 QKV Bias
            if base_attn.qkv.bias is not None:
                old_qkv_b = base_attn.qkv.bias.view(3, self.orig_num_heads, self.head_dim)
                new_qkv_b = old_qkv_b.repeat_interleave(int(expand_ratio), dim=1)
                self.qkv.bias.copy_(new_qkv_b.reshape(-1))

            # --- 处理 Projection 权重 ---
            # 原始 shape: [Dim, H_old * D] -> H 在 dim=1
            old_proj_w = base_attn.proj.weight
            old_proj_w = old_proj_w.view(dim, self.orig_num_heads, self.head_dim)

            # 复制
            new_proj_w = old_proj_w.repeat_interleave(int(expand_ratio), dim=1)

            # 添加噪声
            new_proj_w = new_proj_w + (torch.randn_like(new_proj_w) * noise_std)

            # 赋值
            self.proj.weight.copy_(new_proj_w.reshape(dim, -1))

            if base_attn.proj.bias is not None:
                self.proj.bias.copy_(base_attn.proj.bias)

            print(f"Upcycling initialized: {self.orig_num_heads} -> {self.total_num_heads} heads "
                  f"(Shared: {self.num_shared_heads}, Routed: {self.num_routed_heads})")

    def forward(self, x, condition_embedding=None, is_vfm_condition=False, vfm_teacher_id=None):
        B, N, C = x.shape

        # 1. 计算所有 Head 的 QKV (包含 Shared 和 Routed)
        # 注意：这里维度是 self.total_num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.total_num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H_total, N, D)

        # 2. 统一计算 Attention (并行计算最高效)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v)  # (B, H_total, N, D)

        # ============================================================
        # 【核心逻辑修正】分离 Shared 和 Routed
        # ============================================================

        # A. 拆分
        if self.num_shared_heads > 0:
            # 假设前 N 个是 Shared
            x_shared = x_attn[:, :self.num_shared_heads, :, :]
            x_routed = x_attn[:, self.num_shared_heads:, :, :]
        else:
            x_shared = None
            x_routed = x_attn

        # B. 路由 (只针对 x_routed)
        # 调用高级路由器
        router_logits = self.router(
            x,
            condition_embedding,
            is_vfm_condition=is_vfm_condition,
            routing_mask=None
        )

        # 计算概率 (用于 Aux Loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 选择
        top_k_weights, selected_indices = torch.topk(router_logits, self.num_selected_heads, dim=-1)

        # 构建 Mask (Hard Selection)
        # selected_indices 是针对 routed heads 的索引 (0 ~ num_routed_heads-1)
        mask = F.one_hot(selected_indices, num_classes=self.num_routed_heads).float().sum(dim=2)

        # 扩展 Mask (B, H_routed, N, 1)
        mask_expanded = mask.permute(0, 2, 1).unsqueeze(-1)

        # C. 应用路由结果
        x_routed = x_routed * mask_expanded

        # D. 拼接回 (Shared + Routed)
        if x_shared is not None:
            # 在 Head 维度拼接
            x_final = torch.cat([x_shared, x_routed], dim=1)
        else:
            x_final = x_routed

        # 3. 投影输出
        # x_final: (B, H_total, N, D) -> (B, N, H_total * D)
        x_final = x_final.transpose(1, 2).reshape(B, N, -1)
        x_final = self.proj(x_final)
        x_final = self.proj_drop(x_final)

        # 4. Aux Loss (只基于 Routed 部分)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            importance = router_probs.sum(dim=(0, 1))
            load = mask.sum(dim=(0, 1))
            aux_loss = torch.sum(importance * load) / (B * N)

        # 为了保持接口一致，返回的 indices 可能需要调整或说明，这里返回局部索引
        return x_final, aux_loss, router_probs, selected_indices


class MoHBlock(nn.Module):
    """
    封装 MoHAttention 的 Block
    """

    def __init__(self, base_block, num_selected_heads,
                 condition_dim,  # <--- 新增：必须传入，Router 需要
                 expand_ratio=2,  # <--- 新增：参数化
                 num_shared_heads=4,  # <--- 新增：参数化
                 noise_std=0.02,  # <--- 新增：参数化
                 noisy_gating=True):

        super().__init__()

        # 1. 复制基础 Block 的组件
        self.norm1 = base_block.norm1
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp

        # 复制 LayerScale (如果有)
        self.ls1 = base_block.ls1 if hasattr(base_block, 'ls1') else nn.Identity()
        self.ls2 = base_block.ls2 if hasattr(base_block, 'ls2') else nn.Identity()

        # 复制 DropPath (如果有) - 这一步很重要，防止深层过拟合
        self.drop_path1 = base_block.drop_path1 if hasattr(base_block, 'drop_path1') else nn.Identity()
        self.drop_path2 = base_block.drop_path2 if hasattr(base_block, 'drop_path2') else nn.Identity()

        # 2. 获取维度信息
        # 从 base_block 的 norm 或 attention 中提取 dim
        if hasattr(base_block.norm1, 'normalized_shape'):
            dim = base_block.norm1.normalized_shape[0]
        else:
            dim = base_block.attn.qkv.in_features

        # 3. 初始化 Upcycled MoH Attention
        # 这里的参数现在全部动态传入，不再硬编码
        self.attn = MoHAttention(
            base_attn=base_block.attn,
            dim=dim,
            num_heads=base_block.attn.num_heads,  # 原始头数 (12)
            num_selected_heads=num_selected_heads,
            condition_dim=condition_dim,  # <--- 传入 Router
            expand_ratio=expand_ratio,
            num_shared_heads=num_shared_heads,
            noise_std=noise_std,
            noisy_gating=noisy_gating
        )

    def forward(self, x, condition_embedding=None, is_vfm_condition=False, vfm_teacher_id=None,
                return_expert_outputs=False):

        # 1. Attention 部分 (带路由)
        # 【关键修改】必须将 condition_embedding 传进去，否则 Router 无法根据任务引导
        attn_out, aux_loss, router_probs, top_k_indices = self.attn(
            self.norm1(x),
            condition_embedding=condition_embedding,
            is_vfm_condition=is_vfm_condition,
            vfm_teacher_id=vfm_teacher_id
        )

        # 应用 LayerScale 和 DropPath (Residual Connection)
        x = x + self.drop_path1(self.ls1(attn_out))

        # 2. MLP 部分 (标准)
        # MLP 不需要改动，还是 dense 的
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        # 统一返回接口
        return x, aux_loss, router_probs, top_k_indices, None


class MoH_ViT(nn.Module):
    """
    集成了 MoH (Mixture-of-Heads) 的 Vision Transformer 主干
    """

    def __init__(
            self,
            vit_base_model,
            num_selected_heads=4,  # MoH 特有参数
            moh_layers_indices=None,  # 指定哪些层使用 MoH
            noisy_gating=True,
            out_indices=None,
            **kwargs,
    ):
        super().__init__()
        # --- 1. 复制 ViT 基础属性 (与 Condition_MoE_ViT 相同) ---
        self.embed_dim = vit_base_model.embed_dim
        self.patch_embed = copy.deepcopy(vit_base_model.patch_embed)
        self.pos_embed = copy.deepcopy(vit_base_model.pos_embed)
        self.num_prefix_tokens = getattr(vit_base_model, 'num_prefix_tokens', 1 if hasattr(vit_base_model, 'cls_token') else 0)
        self.pos_drop = copy.deepcopy(getattr(vit_base_model, 'pos_drop', nn.Identity()))
        self.patch_drop = copy.deepcopy(getattr(vit_base_model, 'patch_drop', nn.Identity()))
        self.norm_pre = copy.deepcopy(getattr(vit_base_model, 'norm_pre', nn.Identity()))

        # 处理 CLS/Reg tokens
        if hasattr(vit_base_model, 'cls_token'):
            self.cls_token = copy.deepcopy(vit_base_model.cls_token)
        else:
            self.cls_token = None
        self.reg_token = copy.deepcopy(getattr(vit_base_model, 'reg_token', None))

        self.out_indices = out_indices
        self.dynamic_img_size = getattr(vit_base_model, 'dynamic_img_size', False)
        self.no_embed_class = getattr(vit_base_model, 'no_embed_class', False)

        # --- 2. 构建 Blocks (核心差异) ---
        self.blocks = nn.ModuleList()
        depth = len(vit_base_model.blocks)

        # 如果未指定索引，默认所有层都用 MoH (或者你可以设为空列表)
        if moh_layers_indices is None:
            moh_layers_indices = list(range(depth))

        for i in range(depth):
            base_blk = vit_base_model.blocks[i]
            if i in moh_layers_indices:
                self.blocks.append(MoHBlock(
                    base_block=base_blk,
                    num_selected_heads=num_selected_heads,
                    condition_dim=self.embed_dim,  # <--- 传入 condition_dim (通常等于 embed_dim)
                    expand_ratio=kwargs.get('expand_ratio', 2),  # 从 kwargs 获取配置
                    num_shared_heads=kwargs.get('num_shared_heads', 4),
                    noise_std=0.02,
                    noisy_gating=noisy_gating
                ))
            else:
                # 保持标准 Block
                self.blocks.append(copy.deepcopy(base_blk))

        self.norm = copy.deepcopy(vit_base_model.norm) if hasattr(vit_base_model, 'norm') else nn.Identity()

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

    def forward_intermediate_features(self, x, condition_embedding=None, is_vfm_condition=False, vfm_teacher_id=None,
                                      return_expert_outputs=False):
        # 预处理
        x = self.patch_embed(x)
        if hasattr(self, '_pos_embed'):
            x = self._pos_embed(x)
        else:
            x = x + self.pos_embed  # 简单回退
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        total_aux_loss_agg = torch.tensor(0.0, device=x.device)
        gate_regularization_loss = torch.tensor(0.0, device=x.device)  # MoH 暂无 gate reg，除非加 L2

        output = []
        gate_list = []
        topk_indices_list = []

        for i, blk in enumerate(self.blocks):
            if isinstance(blk, MoHBlock):
                # MoH Block 调用
                x, aux_loss, router_probs, top_k_indices, _ = blk(
                    x, condition_embedding, is_vfm_condition, vfm_teacher_id
                )
                total_aux_loss_agg += aux_loss
                if router_probs is not None: gate_list.append(router_probs)
                if not self.training: topk_indices_list.append(top_k_indices)
            else:
                # 标准 Block 调用
                x = blk(x)

            # 收集中间层输出
            if self.out_indices and i in self.out_indices:
                x_norm = self.norm(x) if self.norm is not None else x
                output.append(x_norm)

        # 调整 Aux Loss 权重 (通常 MoH 需要一个系数)
        # 外部调用时通常会乘一个系数，这里直接返回 raw sum
        return output, total_aux_loss_agg, gate_list, topk_indices_list, gate_regularization_loss
