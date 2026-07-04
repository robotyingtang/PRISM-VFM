import torch
import torch.nn as nn

from models.build_models import build_model
from losses import DistillLoss
from collections import OrderedDict


class Condition_MoE_MTMT_Distiller(nn.Module):
    """Multi-Task Multi-Teacher Distiller for Conditioned MoE Student"""

    def __init__(
            self,
            img_size: tuple,
            tea_configs: dict,
            stu_config: dict,
            loss_type: str = "cos+l1",
            task_out: bool = False,
            stu_criterion: nn.Module = None,
            dataname: str = None,
            tasks: list = None,
            task_specific_teacher_configs: dict = None,
            task_specific_distill_loss_type: str = "mse",
    ) -> None:
        super().__init__()

        # 蒸馏损失函数 (用于VFM教师 和 可选的任务特定教师)
        self.kd_criterion = DistillLoss(loss_type)
        if task_specific_teacher_configs:
            self.task_specific_kd_criterion = DistillLoss(task_specific_distill_loss_type)
        else:
            self.task_specific_kd_criterion = None

        # 任务相关设置 (主要用于阶段2)
        self.task_out = task_out  # task_out 现在更多地指示是否计算任务损失
        if self.task_out:
            assert stu_criterion is not None, "Student criterion (for task loss) must be provided if task_out is True."
            assert tasks is not None, "Tasks list must be provided if task_out is True."
            self.stu_criterion = stu_criterion
            self.tasks = tasks  # 这里的tasks可能是任务名称列表，需要映射到task_ids (0, 1, ...)

        # --- 构建教师模型 ---
        # 1. VFM Teachers (用于阶段1)
        self.teachers = nn.ModuleDict()
        tea_dims = OrderedDict()
        vfm_projection_configs_for_student = {}  # 用于配置学生端的VFM投影头

        for i, (tea_name_key, tea_config) in enumerate(tea_configs.items()):
            tea_model = build_model(arch="backbone", img_size=img_size, backbone_args=tea_config) # 您的构建函数
            # Freeze teacher
            tea_model.eval()
            for param in tea_model.parameters():
                param.requires_grad = False

            internal_teacher_key = str(i)
            self.teachers[internal_teacher_key] = tea_model


            if hasattr(tea_model.model, "embed_dim"):
                tea_dims[internal_teacher_key] = tea_model.model.embed_dim
            elif hasattr(tea_model.model, "config"):
                tea_dims[internal_teacher_key] = tea_model.model.config.hidden_size
            else:
                raise RuntimeError
            print(tea_name_key, "->", tea_dims[internal_teacher_key])

        # 将从教师配置中推断出的信息加入学生配置
        # stu_config["backbone"]["num_vfm_teachers"] = len(self.vfm_teachers)
        stu_config["backbone"]["tea_dims"] = tea_dims
        if "num_tasks" not in stu_config["backbone"] and tasks:
            stu_config["backbone"]["num_tasks"] = len(tasks)

        # `task_head_configs` 应该由外部根据任务定义好并传入 stu_config["backbone"]
        # 例如: stu_config["backbone"]["task_head_configs"] = {"0": {"out_features": 10}, ...}

        # arch="mt" 或 "backbone" 的判断可能需要调整，或者 build_model 更智能
        if task_out:  # for s2
            # multi-task model
            self.student = build_model(
                arch="mt",
                img_size=img_size,
                backbone_args=stu_config["backbone"],
                dataname=dataname,
                tasks=tasks,
            )
            self.vfm_training = stu_config["backbone"]["vfm_training"]  # task阶段不进行VFM蒸馏

        else:
            # backbone only
            self.student = build_model(
                arch="backbone",
                img_size=img_size,
                backbone_args=stu_config["backbone"],
            )


    def forward(
            self,
            batch,
            step=-1,
            # vfm_training: bool = True,  # 是否在VFM蒸馏阶段
            # task_training: bool = False,  # 是否在任务训练阶段 (如果有任务输出)
            ):  # 新增参数
        # images, task_gts 从 batch 中提取
        self.student.train()

        ########################## 处理 batch 的不同格式 ##################################
        if isinstance(batch, dict):
            images = batch["image"]
            # print("train batch image:", batch["image"].shape)
            # print("model img_size:", self.student.backbone.model.patch_embed.img_size)
            task_gts = batch["label"] # task_gts 用于计算任务损失，不是直接给学生的条件ID
            # if self.task_out:  # 任务训练阶段
            #     ids_for_student = batch.get("task_id", None)#  它们应该是 (B,) 的张量
            #     # if ids_for_student is None:
            #     #     raise ValueError("task_out is True, but 'task_id' tensor is missing from batch dict.")
            # else:  # VFM 蒸馏阶段
            ids_for_student = batch.get("vfm_teacher_id", None)
                # if ids_for_student is None:
                #     raise ValueError("VFM distillation, but 'vfm_teacher_id' tensor is missing from batch dict.")
            student_input_batch = [images, ids_for_student]  # 学生期望的格式

        elif isinstance(batch, list) or isinstance(batch, tuple):
            # 如果是列表/元组，假设 batch = [images, ids_for_student_condition]
            # ids_for_student_condition 应该是 (B,) 的 vfm_teacher_ids 或 task_ids
            images = batch[0]
            ids_for_student = batch[1] if len(batch) > 1 else None
            # if ids_for_student is None:
            #     raise ValueError(
            #         "If batch is a list/tuple, it must contain images and ids (vfm_teacher_id or task_id).")
            student_input_batch = batch  # 直接传递
            task_gts = batch[2] if len(batch) > 2 else None  # 如果GTs也在列表里

        else:  # 纯图像，仅用于无条件推理或非常简单的场景
            images = batch
            ids_for_student = None  # 这种情况下，学生模型可能进入 default/inference 路径
            student_input_batch = images  # 学生模型只收到图像
            task_gts = None

        ###################################################################################################
        tea_feas_dict = OrderedDict()  # 用于存储教师输出
        # 只有在VFM蒸馏阶段才需要教师输出 (即 not self.task_out)
        # if not self.task_out:
        with torch.no_grad():
            # 遍历 self.teachers 字典的键 (假设这些键是 "0", "1", ... 或可以映射到索引)
            # 或者直接迭代唯一的 vfm_teacher_ids 出现在当前批次中
            # unique_teacher_ids_in_current_batch = torch.unique(vfm_teacher_ids_from_batch)

            for i, (tea_name, tea_model) in enumerate(self.teachers.items()):
                teacher_id_tensor = torch.tensor(i, device=images.device)  # 假设教师名称是 "0", "1", ...
                teacher_id_str = str(i)  # "0", "1", ...

                # 1. 创建掩码，选择属于当前 teacher_id_str 的样本
                mask_for_current_teacher = (ids_for_student == teacher_id_tensor)
                if torch.sum(mask_for_current_teacher) == 0:  # 理论上不会发生，因为我们遍历的是unique_ids
                    continue

                # 2. 获取只属于当前教师的图像子集
                images_for_this_teacher = images[mask_for_current_teacher] # images_for_this_teacher: (B_masked, C, H, W)

                # 3. 当前教师模型只处理这个子集
                teacher_output_for_masked_batch = tea_model(images_for_this_teacher)  # teacher_output_for_masked_batch: 特征列表或单个特征张量
                tea_feas_dict[teacher_id_str] = teacher_output_for_masked_batch # 其批次维度现在是 B_masked


        if self.task_out:  # 根据训练阶段调用学生模型    # 对应 Condition_MoE_PRISM 的 task_training=True
            student_outputs_dict = self.student(student_input_batch, vfm_training=self.vfm_training, task_training=True)
            # student_outputs_dict 结构: {"aux_loss": ..., "task_outputs": {"task_idx_str": logits_B_masked, ...}}
            # task_loss_dict = self.stu_criterion(student_outputs_dict["output_for_tasks"], task_gts, self.tasks)
            aux_loss_student = student_outputs_dict.get("aux_loss", torch.tensor(0.0, device=images.device))
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))
            # print(f"Aux Loss Student: {aux_loss_student.item()}")
            kd_loss_dict = None
            if self.vfm_training:

                aligned_feas_dict = student_outputs_dict["vfm_student_projections"]

                kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)
            task_loss_dict = self.stu_criterion(student_outputs_dict["output_for_tasks"], task_gts, self.tasks)

            # ### 关闭 gate_entropy_loss ##########
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))

            return kd_loss_dict, 0.1 * aux_loss_student, task_loss_dict# 在任务阶段，通常不返回kd_loss，除非你也做教师指导的任务学习

        else:  # 对应 Condition_MoE_PRISM 的 vfm_training=True
            student_outputs_dict = self.student(student_input_batch, vfm_training=True, task_training=False) # student_outputs_dict 结构: {"vfm_student_projections": {"tea_idx_str": [feature_map_B_masked_CHW], ...}, "aux_loss": ...}

            aligned_feas_dict = student_outputs_dict["vfm_student_projections"]
            aux_loss = student_outputs_dict.get("aux_loss", torch.tensor(0.0, device=images.device))
            gate_regularization_loss = student_outputs_dict.get("gate_regularization_loss", torch.tensor(0.0, device=images.device))
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))

            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict, step=step)

            # print(f"KD Loss Dict: {kd_loss_dict}")
            # if "aux_loss" in student_outputs_dict:
            #     total_loss_dict["aux_loss"] = output_dict["aux_loss"]
            # total_loss_dict["kd_loss"] = kd_loss_dict

            # #### 关闭 gate_entropy_loss ##########
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))

            # print(f"Aux Loss: {aux_loss.item()}, Gate Reg Loss: {0.1 * gate_regularization_loss.item()}")
            # print()
            return kd_loss_dict, 0.1 * aux_loss, gate_regularization_loss




    def forward_val(self, batch):  # 验证时需要提供task_ids
        self.student.eval()

        ########################## 处理 batch 的不同格式 ##################################
        if isinstance(batch, dict):
            images = batch["image"]
            # print("val batch image:", batch["image"].shape)
            task_gts = batch["label"]  # task_gts 用于计算任务损失，不是直接给学生的条件ID
            ids_for_student = batch.get("vfm_teacher_id", None)
            student_input_batch = [images, ids_for_student]  # 学生期望的格式

        elif isinstance(batch, list) or isinstance(batch, tuple):
            # 如果是列表/元组，假设 batch = [images, ids_for_student_condition]
            images = batch[0]
            ids_for_student = batch[1] if len(batch) > 1 else None
            student_input_batch = batch  # 直接传递
            task_gts = batch[2] if len(batch) > 2 else None  # 如果GTs也在列表里

        else:  # 纯图像，仅用于无条件推理或非常简单的场景
            images = batch
            ids_for_student = None  # 这种情况下，学生模型可能进入 default/inference 路径
            student_input_batch = images  # 学生模型只收到图像
            task_gts = None

        ###################################################################################################
        tea_feas_dict = OrderedDict()  # 用于存储教师输出

        with torch.no_grad():
            # 遍历 self.teachers 字典的键 (假设这些键是 "0", "1", ... 或可以映射到索引)
            # 或者直接迭代唯一的 vfm_teacher_ids 出现在当前批次中
            # unique_teacher_ids_in_current_batch = torch.unique(vfm_teacher_ids_from_batch)

            for i, (tea_name, tea_model) in enumerate(self.teachers.items()):
                teacher_id_tensor = torch.tensor(i, device=images.device)  # 假设教师名称是 "0", "1", ...
                teacher_id_str = str(i)  # "0", "1", ...

                # 1. 创建掩码， 选择属于当前 teacher_id_str 的样本
                mask_for_current_teacher = (ids_for_student == teacher_id_tensor)
                if torch.sum(mask_for_current_teacher) == 0:  # 理论上不会发生，因为我们遍历的是unique_ids
                    continue

                # 2. 获取只属于当前教师的图像子集
                images_for_this_teacher = images[
                    mask_for_current_teacher]  # images_for_this_teacher: (B_masked, C, H, W)

                # 3. 当前教师模型只处理这个子集
                teacher_output_for_masked_batch = tea_model(images_for_this_teacher)  # teacher_output_for_masked_batch: 特征列表或单个特征张量
                tea_feas_dict[teacher_id_str] = teacher_output_for_masked_batch  # 其批次维度现在是 B_masked


        ######################################################################################################

        if self.task_out:  # 根据训练阶段调用学生模型    # 对应 Condition_MoE_PRISM 的 task_training=True
            student_outputs_dict = self.student(student_input_batch, vfm_training=self.vfm_training, task_training=True)
            aux_loss_student = student_outputs_dict.get("aux_loss", torch.tensor(0.0, device=images.device))
            # gate_regularization_loss = student_outputs_dict.get("gate_regularization_loss", torch.tensor(0.0, device=images.device))
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))
            kd_loss_dict = None
            if self.vfm_training:
                aligned_feas_dict = student_outputs_dict["vfm_student_projections"]

                kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)
            # task_loss_dict = self.stu_criterion(student_outputs_dict["output_for_tasks"], task_gts, self.tasks)
            #     aux_loss_student
            return kd_loss_dict, 0.5 * aux_loss_student, student_outputs_dict["output_for_tasks"]  # 在任务阶段，通常不返回kd_loss，除非你也做教师指导的任务学习

        else:  # 对应 Condition_MoE_PRISM 的 vfm_training=True
            student_outputs_dict = self.student(student_input_batch, vfm_training=True, task_training=False)  # student_outputs_dict 结构: {"vfm_student_projections": {"tea_idx_str": [feature_map_B_masked_CHW], ...}, "aux_loss": ...}
            aux_loss_student = student_outputs_dict.get("aux_loss", torch.tensor(0.0, device=images.device))
            # gate_regularization_loss = student_outputs_dict.get("gate_regularization_loss", torch.tensor(0.0, device=images.device))
            # gate_entropy_loss = student_outputs_dict.get("gate_entropy_loss", torch.tensor(0.0, device=images.device))
            aligned_feas_dict = student_outputs_dict["vfm_student_projections"]

            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)

            return kd_loss_dict, 0.1 * aux_loss_student

# 主要变化和解释 (与您提供的 MTMT_Distiller 对比)：
# 	1. 类名: 改为 MTMT_Condition_MoE_Distiller 以反映其适配新的学生模型。
# 	2. __init__:
# 		○ 教师构建: 分别构建 self.vfm_teachers (用于阶段1) 和可选的 self.task_specific_teachers (用于阶段2)。
# 			§ 教师模型现在可以是简单的lambda函数或nn.Sequential作为占位符，您需要用build_model替换。
# 			§ 重要: 为教师模型添加了 is_token_level 配置，并据此调整了占位符教师的输出形状。您需要确保实际的教师模型能输出token级特征，并且其token数量能与学生对齐。
# 			§ 为学生模型准备了 vfm_projection_configs_for_student，以便学生可以创建正确的投影头。
# 		○ 学生构建:
# 			§ 断言 stu_config["backbone"]["backbone_type"] == "condition_moe_prism"。
# 			§ 将从教师配置中推断的信息（如 num_vfm_teachers, vfm_projection_configs）传递给学生模型的配置。
# 			§ 调用 build_model 构建学生模型（假设 build_model 已更新）。
# 		○ 损失函数: self.kd_criterion 重命名为 self.vfm_kd_criterion，并为任务特定蒸馏添加了 self.task_specific_kd_criterion。
# 	3. forward 方法:
# 		○ 新增参数: current_phase: str (必需), task_ids=None, vfm_teacher_ids=None (必需，取决于阶段)。
# 		○ 学生模型调用: self.student(images, task_ids=task_ids, vfm_teacher_ids=vfm_teacher_ids)。学生模型现在接收条件ID。
# 		○ MoE辅助损失: 从学生输出中获取 aux_loss 并加入到总损失中。
# 		○ 阶段1 (VFM蒸馏):
# 			§ 从 student_outputs 中获取 vfm_student_projections_tokens。
# 			§ 教师模型 (self.vfm_teachers) 也需要输出token级别的特征。
# 			§ self.vfm_kd_criterion 在这些token级特征上计算损失。
# 			§ 返回的字典包含 vfm_kd_loss 和 aux_loss。
# 		○ 阶段2 (任务学习):
# 			§ 从 student_outputs 中获取 task_outputs_tokens。
# 			§ self.stu_criterion 计算任务损失。这里需要根据任务类型，从task_outputs_tokens中提取适合计算损失的预测值（例如，CLS token的输出用于分类）。我在代码中加入了一个示例性的处理。
# 			§ 如果启用了任务特定蒸馏，则类似地计算。
# 			§ 返回的字典包含 task_loss, aux_loss, 和可选的 task_specific_kd_loss。
# 		○ 返回结构: forward 方法现在返回一个包含各种损失项的字典，方便训练循环处理。
# 	4. forward_val 方法:
# 		○ 接收 task_ids 以便学生模型可以被正确条件化。
# 		○ 从 student_outputs 中提取任务相关的预测值（通常是聚合后的，例如CLS token的输出或平均池化）用于评估。
# 		○ 返回聚合后的任务输出和辅助损失。
