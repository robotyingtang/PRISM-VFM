import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import global_print

from einops import rearrange

# Loss functions and hyperparameters
PASCAL_LOSS_CONFIG = {
    "semseg": {"loss_function": "CELoss", "weight": 1},
    "human_parts": {"loss_function": "CELoss", "weight": 2},
    "normals": {
        "loss_function": "L1Loss",
        "parameters": {"normalize": True},
        "weight": 10,
    },
    "sal": {"loss_function": "CELoss", "parameters": {"balanced": True}, "weight": 5},
    "edge": {
        "loss_function": "BalancedBCELoss",
        "parameters": {"pos_weight": 0.95},
        "weight": 50,
    },
}

NYUD_LOSS_CONFIG = {
    "semseg": {"loss_function": "CELoss", "weight": 1},
    "normals": {
        "loss_function": "L1Loss",
        "parameters": {"normalize": True},
        "weight": 10, #10
    },
    "edge": {
        "loss_function": "BalancedBCELoss",
        "parameters": {"pos_weight": 0.95},
        "weight": 50,
    },
    "depth": {"loss_function": "L1Loss", "weight": 1}, #1
}

class BalancedBCELoss(nn.Module):
    # Edge Detection

    def __init__(self, pos_weight=0.95, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        mask = label != self.ignore_index
        masked_output = torch.masked_select(output, mask)  # 1-d tensor
        masked_label = torch.masked_select(label, mask)  # 1-d tensor

        # pos weight: w, neg weight: 1-w
        w = torch.tensor(self.pos_weight, device=output.device)
        factor = 1.0 / (1 - w)
        loss = F.binary_cross_entropy_with_logits(
            masked_output, masked_label, pos_weight=w * factor
        )
        loss /= factor

        return loss


class CELoss(nn.Module):
    # Semantic Segmentation, Human Parts Segmentation, Saliency Detection

    def __init__(self, balanced=False, ignore_index=255):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.balanced = balanced

    def forward(self, output, label):
        label = torch.squeeze(label, dim=1).long()

        if self.balanced:
            mask = label != self.ignore_index
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary

            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            pos_weight = num_labels_neg / num_total
            class_weight = torch.stack((1.0 - pos_weight, pos_weight), dim=0)
            loss = F.cross_entropy(
                output,
                label,
                weight=class_weight,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
        else:
            loss = F.cross_entropy(
                output, label, ignore_index=self.ignore_index, reduction="sum"
            )

        n_valid = (label != self.ignore_index).sum()
        loss /= max(n_valid, 1)

        return loss

class OhemCELoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super(OhemCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)

    def forward(self, output, label):
        label = torch.squeeze(label, dim=1).long()   # [B,H,W]

        # per-pixel CE
        loss = F.cross_entropy(
            output,
            label,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # [B,H,W]

        with torch.no_grad():
            prob = F.softmax(output, dim=1)  # [B,C,H,W]

            valid_mask = (label != self.ignore_index)
            tmp_label = label.clone()
            tmp_label[~valid_mask] = 0

            gt_prob = prob.gather(1, tmp_label.unsqueeze(1)).squeeze(1)  # [B,H,W]
            valid_gt_prob = gt_prob[valid_mask]

            if valid_gt_prob.numel() == 0:
                return loss[valid_mask].mean() * 0.0

            sorted_prob, _ = torch.sort(valid_gt_prob)
            if sorted_prob.numel() < self.min_kept:
                threshold = self.thresh
            else:
                threshold = max(self.thresh, sorted_prob[self.min_kept - 1].item())

            hard_mask = valid_mask & (gt_prob <= threshold)

        hard_loss = loss[hard_mask]
        if hard_loss.numel() == 0:
            hard_loss = loss[valid_mask]

        return hard_loss.mean()
class L1Loss(nn.Module):
    # Normals Estimation, Depth Estimation

    def __init__(self, normalize=False, ignore_index=255):
        super(L1Loss, self).__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if self.normalize:
            # Normalize to unit vector
            output = F.normalize(output, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        masked_output = torch.masked_select(output, mask)
        masked_label = torch.masked_select(label, mask)

        loss = F.l1_loss(masked_output, masked_label, reduction="sum")
        n_valid = torch.sum(mask).item()
        loss /= max(n_valid, 1)

        return loss


class SWMDMSLoss(nn.Module):
    """
    Spatially-Weighted MDMS Loss "Worker" Module.

    Calculates the SW-MDMS loss between a single pair of teacher and student
    feature maps, assuming they are already resized to the same dimensions.

    Args:
        spatial_sigma (float): The sigma value for the Gaussian spatial weights.
        distmat_margin (float): The margin `m2` for the ReLU function.
    """

    def __init__(self, sigma_percentage: float=0.1, distmat_margin: float=0.25):
        super(SWMDMSLoss, self).__init__()
        # if spatial_sigma <= 0:
        #     raise ValueError("spatial_sigma must be positive.")
        self.sigma_percentage = sigma_percentage  # Will be scaled by min(h, w) dynamically
        # self.spatial_sigma = spatial_sigma
        self.distmat_margin = distmat_margin

        # Buffer to cache the spatial weight matrix for efficiency
        self.register_buffer('spatial_weights', None)

    def _create_spatial_weights(self, h, w, device):
        """Helper to create the (N, N) spatial Gaussian weight matrix."""
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                            indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2).float()
        delta = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist_sq = torch.sum(delta ** 2, dim=-1)
        weights = torch.exp(-dist_sq / (2 * self.spatial_sigma ** 2))
        return weights

    def forward(self, stu_fea, tea_fea):
        """
        Args:
            stu_fea (torch.Tensor): Student feature map of shape (b, c, h, w).
            tea_fea (torch.Tensor): Teacher feature map of shape (b, c, h, w).
        Returns:
            torch.Tensor: A single scalar loss value.
        """
        assert stu_fea.shape == tea_fea.shape, "Input feature maps must have the same shape."
        assert len(stu_fea.shape) == 4, "Inputs must be 4D feature maps."

        b, c, h, w = stu_fea.shape

        # --- Create or Update Spatial Weights Cache ---
        if self.spatial_weights is None or self.spatial_weights.shape[0] != h * w:
            self.spatial_sigma = self.sigma_percentage * min(h, w)
            self.spatial_weights = self._create_spatial_weights(h, w, device=stu_fea.device)

        # --- Prepare Features ---
        stu_flat = rearrange(stu_fea, 'b c h w -> b c (h w)')
        tea_flat = rearrange(tea_fea, 'b c h w -> b c (h w)')
        stu_norm = F.normalize(stu_flat, dim=1)
        tea_norm = F.normalize(tea_flat, dim=1)

        # --- Compute Internal Cosine Similarity Matrices ---
        stu_cos_sim = torch.einsum('bci,bcj->bij', stu_norm, stu_norm)
        tea_cos_sim = torch.einsum('bci,bcj->bij', tea_norm, tea_norm)

        # --- Calculate Spatially-Weighted MDMS Loss ---
        diff = torch.abs(stu_cos_sim - tea_cos_sim)
        relu_diff = F.relu(diff - self.distmat_margin)

        # Apply spatial weights and compute the final loss
        weighted_diff_sum = (relu_diff * self.spatial_weights).sum(dim=[1, 2])
        loss_per_item = weighted_diff_sum / self.spatial_weights.sum()
        final_loss = loss_per_item.mean()

        return final_loss





class SAMRelationalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 只比较相邻的一个小窗口，比如 3x3 或 5x5

    def forward(self, stu_fea: torch.Tensor, sam_fea: torch.Tensor):
        b, c, h, w = stu_fea.shape

        # 将特征归一化
        stu_norm = F.normalize(stu_fea, dim=1)
        sam_norm = F.normalize(sam_fea, dim=1)

        # 将图像稍微平移1个像素（向上、向下、向左、向右）
        # 计算每个像素与其右边像素的余弦相似度
        stu_sim_x = torch.sum(stu_norm[:, :, :, :-1] * stu_norm[:, :, :, 1:], dim=1)
        sam_sim_x = torch.sum(sam_norm[:, :, :, :-1] * sam_norm[:, :, :, 1:], dim=1)

        stu_sim_y = torch.sum(stu_norm[:, :, :-1, :] * stu_norm[:, :, 1:, :], dim=1)
        sam_sim_y = torch.sum(sam_norm[:, :, :-1, :] * sam_norm[:, :, 1:, :], dim=1)

        # 惩罚他们之间相对关系的差异 (L1 or MSE)
        # 如果 SAM 在这里有边缘 (sam_sim_x 极低)，强制 stu_sim_x 也降低
        loss_x = F.mse_loss(stu_sim_x, sam_sim_x)
        loss_y = F.mse_loss(stu_sim_y, sam_sim_y)

        return loss_x + loss_y
class SpatialHighFreqLoss(nn.Module):
    """
    匹配特征图在空间维度上的高频能量分布，而不强求通道一对一匹配。
    """

    def __init__(self):
        super().__init__()
        # 使用拉普拉斯算子提取全向高频边缘
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)
        self.register_buffer('weight', kernel.view(1, 1, 3, 3))

    def forward(self, stu_fm: torch.Tensor, sam_fm: torch.Tensor) -> torch.Tensor:
        b, c, h, w = stu_fm.shape

        # 1. 计算每个空间位置的整体能量（或者求均值降维到 1 个通道）
        # (b, 1, h, w)
        stu_energy = torch.mean(torch.abs(stu_fm), dim=1, keepdim=True)
        sam_energy = torch.mean(torch.abs(sam_fm), dim=1, keepdim=True)

        # 2. 提取能量图的高频边缘
        # (b, 1, h, w)
        grad_stu = F.conv2d(stu_energy, self.weight, padding=1)
        grad_sam = F.conv2d(sam_energy, self.weight, padding=1)

        # 3. 计算相对 L1 或 MSE，重点是让 Student 在 SAM 认为是边缘的地方也产生波动
        return F.l1_loss(torch.abs(grad_stu), torch.abs(grad_sam))

class FeatureGradientLoss(nn.Module):
    """
    Calculates a spatial gradient matching loss directly on 4D feature maps (b, c, h, w).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('weight_x', None)
        self.register_buffer('weight_y', None)

    def _init_kernels(self, channels: int, device: torch.device):
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=torch.float32)
        kernel_y = torch.tensor([[-1., -2., -1.], [0.,  0.,  0.], [1.,  2.,  1.]], device=device, dtype=torch.float32)
        self.weight_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.weight_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    def forward(self, stu_fm: torch.Tensor, sam_fm: torch.Tensor) -> torch.Tensor:
        b, c, h, w = stu_fm.shape
        if self.weight_x is None or self.weight_x.shape[0] != c:
            self._init_kernels(c, device=stu_fm.device)

        grad_x_stu = F.conv2d(stu_fm, self.weight_x, padding=1, groups=c)
        grad_y_stu = F.conv2d(stu_fm, self.weight_y, padding=1, groups=c)
        grad_x_sam = F.conv2d(sam_fm, self.weight_x, padding=1, groups=c)
        grad_y_sam = F.conv2d(sam_fm, self.weight_y, padding=1, groups=c)

        mag_stu = torch.abs(grad_x_stu) + torch.abs(grad_y_stu)
        mag_sam = torch.abs(grad_x_sam) + torch.abs(grad_y_sam)

        return F.l1_loss(mag_stu, mag_sam)

class DecorrelationLoss(nn.Module):
    """
    Calculates a decorrelation loss to enforce locality in feature maps.
    This loss penalizes high feature similarity between spatially distant pixels/patches,
    combating the "semantic short-circuiting" problem in ViTs.

    Args:
        sigma_percentage (float): Defines the radius of the "local neighborhood".
                                  Similarity outside this radius will be penalized.
                                  Defaults to 0.1 (10% of the feature map's smaller side).
    """

    def __init__(self, sigma_percentage: float = 0.1):
        super().__init__()
        self.sigma_percentage = sigma_percentage
        self.register_buffer('penalty_mask', None)

    def _create_spatial_weights(self, h, w, device):
        """Helper to create the (N, N) spatial Gaussian weight matrix."""
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                            indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2).float()
        delta = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist_sq = torch.sum(delta ** 2, dim=-1)
        weights = torch.exp(-dist_sq / (2 * self.spatial_sigma ** 2))
        return weights
    def _create_penalty_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Creates a mask that is ~0 for nearby pixels and ~1 for distant pixels.
        This is the inverse of the spatial weights.
        """
        # spatial_weights is a Gaussian matrix where nearby pixels have high values (~1)
        spatial_weights = self._create_spatial_weights(h, w, device)

        # The penalty mask is the inverse: 1.0 - gaussian_weights
        # Now, distant pixels have high values (~1), which means high penalty.
        penalty_mask = 1.0 - spatial_weights
        return penalty_mask

    def forward(self, stu_fea: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stu_fea (torch.Tensor): Student feature map of shape (b, c, h, w).
                                    Note: This can also be the output of unpatchify from a ViT.
        Returns:
            torch.Tensor: A single scalar loss value.
        """
        assert len(stu_fea.shape) == 4, "Input must be a 4D feature map (b, c, h, w)."
        b, c, h, w = stu_fea.shape

        # --- Create or Update Penalty Mask Cache ---
        # Cache the mask for efficiency, as it only depends on feature map size.
        if self.penalty_mask is None or self.penalty_mask.shape[0] != h * w:
            self.spatial_sigma = self.sigma_percentage * min(h, w)
            self.penalty_mask = self._create_penalty_mask(h, w, device=stu_fea.device)

        # --- Prepare Features and Compute Correlation ---
        # Flatten spatial dimensions and normalize along the channel dimension
        stu_flat = rearrange(stu_fea, 'b c h w -> b (h w) c')  # -> (B, N, C)
        stu_norm = F.normalize(stu_flat, dim=-1)  # Normalize each patch/pixel vector

        # Compute the internal cosine similarity matrix for each item in the batch
        # This gives us the correlation matrix.
        # stu_corr[b, i, j] = similarity between pixel i and pixel j in sample b
        student_corr = torch.bmm(stu_norm, stu_norm.transpose(1, 2))  # -> (B, N, N)

        # --- Calculate the Loss ---
        # We are only interested in the absolute correlation, as high negative correlation
        # is also a form of long-range relationship we want to suppress in shallow layers.
        abs_student_corr = torch.abs(student_corr)

        # Apply the penalty mask element-wise.
        # This effectively zeroes out the correlation between nearby pixels
        # and keeps the correlation value for distant pixels.
        penalized_corr = abs_student_corr * self.penalty_mask

        # The loss is the mean of these penalized correlation values.
        # We want to drive all long-range correlations to zero.
        loss = penalized_corr.mean()

        return loss
def get_loss_functions(task_loss_config):
    """
    Get loss function for each task
    """

    key2loss = {
        "CELoss": CELoss,
        "OhemCELoss": OhemCELoss,
        "BalancedBCELoss": BalancedBCELoss,
        "L1Loss": L1Loss,
    }

    # Get loss function for each task
    loss_fx = key2loss[task_loss_config["loss_function"]]
    if "parameters" in task_loss_config:
        loss_ft = loss_fx(**task_loss_config["parameters"])
    else:
        loss_ft = loss_fx()

    return loss_ft


class MultiTaskLoss(nn.Module):
    """
    Multi-Task loss with different loss functions and weights
    """

    def __init__(self, tasks, loss_ft, loss_weights):
        super(MultiTaskLoss, self).__init__()
        assert set(tasks) == set(loss_ft.keys())
        assert set(tasks) == set(loss_weights.keys())
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, tasks):
        # for k,v in self.loss_weights.items():
        #     global_print(f"Loss weight for {k}")
        # print("--------------")
        # for k,v in self.loss_ft.items():
        #     global_print(f"Loss function for {k}")
        #
        # print("--------------")
        # for k,v in pred.items():
        #     global_print(f"Prediction for {k}")
        # print("--------------")
        # for k,v in gt.items():
        #     global_print(f"Ground truth for {k}")
        # print("--------------")
        # global_print(tasks)
        out = {t: self.loss_weights[t] * self.loss_ft[t](pred[t], gt[t]) for t in tasks}
        out["total"] = torch.sum(torch.stack([out[t] for t in tasks]))

        return out


def get_criterion(dataname, tasks):
    if dataname == "pascalcontext":
        losses_config = PASCAL_LOSS_CONFIG
    elif dataname == "nyud":
        losses_config = NYUD_LOSS_CONFIG
    else:
        raise NotImplementedError

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss_functions(losses_config[task]) for task in tasks}
    )
    loss_weights = {task: losses_config[task]["weight"] for task in tasks}

    return MultiTaskLoss(tasks, loss_ft, loss_weights)

class DistillLoss(nn.Module):
    """
    Distillation loss with dynamic weight annealing for a specific teacher.
    """

    def __init__(self, loss_type,
                 sam_teacher_name='2',
                 sam_weight_schedule={
                    'start_weight': 1.0,  # Initial weight
                    'end_weight': 1.0,    # Final weight
                    'start_step': 20016,  # Step to start increasing weight
                    'end_step': 28022,     # Step to finish increasing weight
                }):
        """
        Args:
            loss_type (str): The type of loss to use.
            sam_teacher_name (str, optional): The name of the SAM teacher (e.g., '2').
                                              Losses from this teacher will be weighted.
            sam_weight_schedule (dict, optional): Configuration for the weight schedule.
                Example: {
                    'start_weight': 0.1,  # Initial weight
                    'end_weight': 1.0,    # Final weight
                    'start_step': 25000,  # Step to start increasing weight
                    'end_step': 50000     # Step to finish increasing weight
                }
        """
        super(DistillLoss, self).__init__()
        self.loss_type = loss_type

        # --- 新增代码：保存权重调度配置 ---
        self.sam_teacher_name = sam_teacher_name
        self.sam_weight_schedule = sam_weight_schedule
        # ------------------------------------

        if "l1" in loss_type:
            self.l1_loss = nn.SmoothL1Loss()
        if "cos" in loss_type:
            self.cos_loss = nn.CosineEmbeddingLoss()
            self.cos_target = torch.tensor(1.0, requires_grad=False)
        if loss_type == "l2":
            self.l2_loss = nn.MSELoss()
        if "swmdms" in loss_type:
            self.swmdms_loss = SWMDMSLoss(sigma_percentage=0.1, distmat_margin=0.25)
        if "decorrelation" in loss_type:
            self.decorrelation_loss = DecorrelationLoss(sigma_percentage=0.1)
            self.decorrelation_layers = [0, 1] #0, 1
            self.decorrelation_lambda = 0.5
        # --- 新增：SAM 梯度损失初始化与独立权重 (mu) ---
        if "sam_grad" in loss_type:
            self.sam_grad_loss = SAMRelationalLoss() #SpatialHighFreqLoss()  #或者 FeatureGradientLoss()，根据你想要的具体实现选择
            self.sam_grad_mu = 5.0  # 建议初始设为 0.5，视训练情况调整
            self.sam_grad_layers = [0, 1]
        # 更新支持的 base 检查
        supported_bases = ["cos", "l1", "l2", "swmdms", "decorrelation", "sam_grad"]
        if not any(base in loss_type for base in supported_bases):
            raise NotImplementedError

    def _calculate_sam_weight(self, step):
        """Helper function to calculate the current weight for the SAM loss."""
        if self.sam_teacher_name is None or self.sam_weight_schedule is None or step < 0:
            return 1.0

        cfg = self.sam_weight_schedule
        start_w, end_w = cfg['start_weight'], cfg['end_weight']
        start_s, end_s = cfg['start_step'], cfg['end_step']

        if step < start_s:
            return start_w
        if step > end_s:
            return end_w

        # Linear interpolation
        progress = (step - start_s) / (end_s - start_s)
        current_weight = start_w + progress * (end_w - start_w)
        return current_weight

    def forward(self, tea_feas_dict, aligned_feas_dict, step=-1):
        """
        Calculates the distillation loss.

        Args:
            tea_feas_dict (dict): Dictionary of teacher features.
            aligned_feas_dict (dict): Dictionary of student features aligned to teachers.
            step (int): The current training step, used for weight annealing.

        Returns:
            dict: A dictionary containing all calculated losses, including the total.
        """
        out = {}

        # 1. 在 forward 开始时根据当前 step 计算 SAM 教师的权重
        current_sam_weight = self._calculate_sam_weight(step)

        # (可选) 如果你想在 TensorBoard 中监控这个权重，可以这样做：
        # out['sam_loss_weight'] = torch.tensor(current_sam_weight) # 需要在训练循环中处理这个非 loss 值

        # 2. 计算去相关性损失 (这部分逻辑保持不变)
        if "decorrelation" in self.loss_type:
            some_teacher_name = next(iter(aligned_feas_dict))
            stu_fea_list = aligned_feas_dict[some_teacher_name]

            total_decorr_loss = 0.0
            for i in self.decorrelation_layers:
                stu_fea_tokens = stu_fea_list[i]
                total_decorr_loss += self.decorrelation_loss(stu_fea_tokens)

            out["decorrelation_loss"] = self.decorrelation_lambda * total_decorr_loss

        # 3. 遍历所有教师来计算蒸馏损失
        for tea_name in tea_feas_dict.keys():
            # 确定当前教师的损失权重
            loss_weight = 1.0
            if self.sam_teacher_name is not None and tea_name == self.sam_teacher_name:
                loss_weight = current_sam_weight

            tea_fea_list = tea_feas_dict[tea_name]
            stu_fea_list = aligned_feas_dict[tea_name]

            # 遍历每个 level 的特征
            for i, (tea_fea, stu_fea) in enumerate(zip(tea_fea_list, stu_fea_list)):
                assert len(tea_fea.shape) == len(stu_fea.shape)

                # 特征图尺寸对齐 (逻辑保持不变)
                if tea_fea.shape[2:] != stu_fea.shape[2:]:
                    if tea_fea.shape[2] > stu_fea.shape[2]:
                        stu_fea = F.interpolate(
                            stu_fea, size=tea_fea.shape[2:], mode="bilinear", align_corners=False
                        )
                    else:
                        tea_fea = F.interpolate(
                            tea_fea, size=stu_fea.shape[2:], mode="bilinear", align_corners=False
                        )
                assert tea_fea.shape == stu_fea.shape

                # 计算基础损失值
                # loss_value = 0.0
                loss_value = torch.tensor(0.0, device=stu_fea.device)  # 依然建议加上这层双保险

                # 【核心修复】：将严格匹配改为特征子串匹配
                # 只要名字里包含 "cos" 和 "l1"（且不是 swmdms 的特殊情况），就计算 cos+l1
                if "cos" in self.loss_type and "l1" in self.loss_type and "swmdms" not in self.loss_type:
                    l1_loss = self.l1_loss(stu_fea, tea_fea)
                    target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
                    cos_loss = self.cos_loss(stu_fea.flatten(1), tea_fea.flatten(1), target)
                    loss_value = 0.9 * cos_loss + 0.1 * l1_loss
                    # global_print("cos+l1")
                # 仅包含 "cos" 不包含 "l1"
                elif "cos" in self.loss_type and "l1" not in self.loss_type:
                    target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
                    loss_value = self.cos_loss(stu_fea.flatten(1), tea_fea.flatten(1), target)

                # 包含 "l2"
                elif "l2" in self.loss_type:
                    loss_value = self.l2_loss(stu_fea, tea_fea)

                # 特殊组合 "cos+l1+swmdms"
                elif "cos" in self.loss_type and "l1" in self.loss_type and "swmdms" in self.loss_type:
                    l1_loss = self.l1_loss(stu_fea, tea_fea)
                    target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
                    cos_loss = self.cos_loss(stu_fea.flatten(1), tea_fea.flatten(1), target)
                    swmdms_loss = self.swmdms_loss(stu_fea, tea_fea)
                    loss_value = 0.7 * cos_loss + 0.1 * l1_loss + 0.2 * swmdms_loss

                # --- 新增：专门针对 SAM 教师计算梯度边缘损失 ---
                if "sam_grad" in self.loss_type and tea_name == self.sam_teacher_name and i in self.sam_grad_layers:
                    grad_loss = self.sam_grad_loss(stu_fea, tea_fea)
                    # 将梯度损失加到当前 level 的基础损失上
                    # loss_value = loss_weight * self.sam_grad_mu * grad_loss

                    # (可选) 记录一下单独的梯度损失，方便 TensorBoard 监控
                    out[f"{tea_name}_grad_loss_level{i + 1}"] = loss_weight * self.sam_grad_mu * grad_loss #grad_loss.detach()  # 这个值不参与反向传播，只用于监控

                out[f"{tea_name}_level{i + 1}"] = loss_weight * loss_value
                # 4. 将计算出的损失乘以其对应的权重
                loss_key = f"{tea_name}_level{i + 1}"
                out[loss_key] = loss_weight * loss_value

        # --- 修复：确保只汇总以 'loss' 结尾的标量张量 ---
        # 避免把 out 字典里可能存的其他监控值（如 weight 数值、detach 掉的中间变量）算进反向传播
        # global_print(out.items())
        valid_losses = [v for k, v in out.items() if 'loss' in k or 'level' in k]
        # 过滤掉不需要计算梯度的 detached 项
        # valid_losses = [v for v in valid_losses if v.requires_grad]

        out["total"] = torch.sum(torch.stack(valid_losses))
        # # 5. 将所有已经加权过的损失相加得到总损失 (逻辑保持不变)
        # out["total"] = torch.sum(
        #     torch.stack([out[t] for t in out.keys()]))  # 增加了 'loss' in t 来避免加权非 loss 值

        return out


# class DistillLoss(nn.Module):
#     """
#     Distillation loss
#     """
#
#     def __init__(self, loss_type):
#         super(DistillLoss, self).__init__()
#         self.loss_type = loss_type
#         if loss_type == "cos+l1":
#             self.l1_loss = nn.SmoothL1Loss()
#             self.cos_loss = nn.CosineEmbeddingLoss()
#             self.cos_target = torch.tensor(1.0, requires_grad=False)
#         elif loss_type == "cos":
#             self.cos_loss = nn.CosineEmbeddingLoss()
#             self.cos_target = torch.tensor(1.0, requires_grad=False)
#         elif loss_type == "l2":
#             self.l2_loss = nn.MSELoss()
#         elif loss_type == "cos+l1+swmdms":
#             self.l1_loss = nn.SmoothL1Loss()
#             self.cos_loss = nn.CosineEmbeddingLoss()
#             self.cos_target = torch.tensor(1.0, requires_grad=False)
#
#             # Check if required arguments are provided
#             self.swmdms_loss = SWMDMSLoss(
#                 sigma_percentage=0.1,
#                 distmat_margin=0.25
#             )
#         elif loss_type == "cos+l1+decorrelation":
#             self.l1_loss = nn.SmoothL1Loss()
#             self.cos_loss = nn.CosineEmbeddingLoss()
#             self.cos_target = torch.tensor(1.0, requires_grad=False)
#             self.decorrelation_loss = DecorrelationLoss(
#                 sigma_percentage=0.1
#             )
#             self.decorrelation_layers = [0, 1]  # Specify which layers to apply decorrelation loss
#             self.decorrelation_lambda = 1.0
#         else:
#             raise NotImplementedError
#
#     def forward(self, tea_feas_dict, aligned_feas_dict):
#         out = {}
#
#         if "decorrelation" in self.loss_type:
#             # decorrelation_loss 只与学生特征有关，且通常只在第一个“老师”的特征上计算一次即可
#             # 因为对于不同的老师，学生的基础特征是一样的
#             some_teacher_name = next(iter(aligned_feas_dict))
#             stu_fea_list = aligned_feas_dict[some_teacher_name]
#
#             total_decorr_loss = 0.0
#             for i in self.decorrelation_layers:
#                 # 假设学生特征是 (B, N, D), 需要 unpatchify
#                 stu_fea_tokens = stu_fea_list[i]
#                 # 这里你需要 unpatchify 函数
#                 # stu_fea_map = unpatchify(stu_fea_tokens)
#                 # ---- 临时的 unpatchify 实现 ----
#                 # 假设是方形 patch
#                 # num_patches = stu_fea_tokens.shape[1]
#                 # h = w = int(num_patches ** 0.5)
#                 # stu_fea_map = rearrange(stu_fea_tokens, 'b (h w) c -> b c h w', h=h, w=w)
#                 # ---- 结束临时实现 ----
#
#                 total_decorr_loss += self.decorrelation_loss(stu_fea_tokens)
#
#             # 将加权后的总去相关性损失存入输出字典
#             out["decorrelation_loss"] = self.decorrelation_lambda * total_decorr_loss
#             # print(f"Decorrelation Loss: {out['decorrelation_loss'].item()}")
#         for tea_name in tea_feas_dict.keys():
#             # print(f"Calculating KD loss for {tea_name}...")
#             tea_fea_list = tea_feas_dict[tea_name]
#             stu_fea_list = aligned_feas_dict[tea_name]
#
#             # 4 levels
#             for i, (tea_fea, stu_fea) in enumerate(zip(tea_fea_list, stu_fea_list)):
#
#                 assert len(tea_fea.shape) == len(stu_fea.shape)
#
#                 # Resize to match the larger resolution between teacher and student
#                 if tea_fea.shape[2:] != stu_fea.shape[2:]:
#                     if tea_fea.shape[2] > stu_fea.shape[2]:
#                         stu_fea = F.interpolate(
#                             stu_fea,
#                             size=tea_fea.shape[2:],
#                             mode="bilinear",
#                             align_corners=False,
#                         )
#                     else:
#                         tea_fea = F.interpolate(
#                             tea_fea,
#                             size=stu_fea.shape[2:],
#                             mode="bilinear",
#                             align_corners=False,
#                         )
#                 assert tea_fea.shape == stu_fea.shape
#
#                 if self.loss_type == "cos+l1" or self.loss_type == "cos+l1+decorrelation":
#                     l1_loss = self.l1_loss(stu_fea, tea_fea)
#                     target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
#                     cos_loss = self.cos_loss(
#                         stu_fea.flatten(1), tea_fea.flatten(1), target
#                     )
#                     out[tea_name + "_level" + str(i + 1)] = (
#                         0.9 * cos_loss + 0.1 * l1_loss
#                     )
#                 elif self.loss_type == "cos":
#                     target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
#                     out[tea_name + "_level" + str(i + 1)] = self.cos_loss(
#                         stu_fea.flatten(1), tea_fea.flatten(1), target
#                     )
#                 elif self.loss_type == "l2":
#                     out[tea_name + "_level" + str(i + 1)] = self.l2_loss(
#                         stu_fea, tea_fea
#                     )
#                 elif self.loss_type == "cos+l1+swmdms":
#                     l1_loss = self.l1_loss(stu_fea, tea_fea)
#                     target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
#                     cos_loss = self.cos_loss(
#                         stu_fea.flatten(1), tea_fea.flatten(1), target
#                     )
#                     swmdms_loss = self.swmdms_loss(stu_fea, tea_fea)
#                     out[tea_name + "_level" + str(i + 1)] = (
#                             0.7 * cos_loss + 0.1 * l1_loss + 0.2 * swmdms_loss
#                     )
#
#
#         # Sum up all KD losses
#         out["total"] = torch.sum(torch.stack([out[t] for t in out.keys()]))
#
#         return out

