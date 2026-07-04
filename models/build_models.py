import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils.configs import get_output_num
from models.backbones.model_set import hf_names, timm_names
from models.heads import DecodeHead


def build_model(
    arch: str,
    img_size: tuple,
    backbone_args: dict,
    dataname: str = None,
    tasks: list = None,
) -> nn.Module:
    """
    Initialize the model
    """

    backbone = get_backbone(img_size=img_size, tasks=tasks, **backbone_args)

    if arch == "backbone":
        return backbone
    elif arch == "mt":
        assert dataname is not None and tasks is not None
        heads = get_head(img_size=img_size, embed_dim=backbone.embed_dim, tasks=tasks, dataname=dataname)
        model = MultiTaskModel(backbone, heads, tasks)
        return model
    else:
        raise NotImplementedError


def get_backbone(backbone_type: str, img_size: tuple, tasks: list = None, **args) -> nn.Module:
    """Return backbone"""

    if backbone_type == "condition_moe_prism":
        from models.backbones.condition_moe_prism import Condition_MoE_PRISM

        backbone = Condition_MoE_PRISM(img_size=img_size, tasks=tasks, **args)

    elif backbone_type in timm_names:
        from models.backbones.timm_encoders import TimmEncoder

        backbone = TimmEncoder(backbone_type=backbone_type, img_size=img_size, **args)
    elif backbone_type in hf_names:
        from models.backbones.hf_encoders import HfEncoder

        backbone = HfEncoder(backbone_type=backbone_type, img_size=img_size, **args)
    else:
        raise NotImplementedError

    return backbone


def get_head(img_size: tuple, embed_dim: int, tasks: list, dataname: str, **args) -> nn.ModuleDict:
    """Return heads"""

    heads = nn.ModuleDict()
    for task in tasks:
        heads[task] = DecodeHead(
            img_size=img_size, in_channels=embed_dim, num_classes=get_output_num(task, dataname), **args
        )
    return heads


class MultiTaskModel(nn.Module):
    """Multi-Task model with shared encoder + task-specific heads"""

    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict, tasks: list) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.tasks = tasks


    def forward(self, x: torch.Tensor,vfm_training=False, task_training=False) -> dict:
        out = {}
        if self.backbone.__class__.__name__ == "Condition_MoE_PRISM":

            img_size = x[0].size()[-2:]

            output = self.backbone(x, vfm_training=vfm_training, task_training=True)
            # aligned_feas_dict, out_feas_dict = self.backbone(x)
            ## aligned_feas_dict: dict of {tea_name: list of [B, C_T, H, W]}
            ## out_feas_dict: dict of {task: list of [B, C, H, W]}
            feature_for_tasks = output["feature_for_tasks"]
            for task in self.tasks:
                inter_in=self.heads[task](feature_for_tasks[task])
                #print(f"inter_in shape: {inter_in.shape}")
                # print(img_size)
                out[task] = F.interpolate(inter_in, img_size, mode="bilinear")
            output["output_for_tasks"] = out
            return output

        else:
            output = {}
            img_size = x[0].size()[-2:]

            batch = x
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
            # img_size = x.size()[2:]
            encoder_output = self.backbone(images)
            # print(f"encoder_output shape: {encoder_output.shape}")
            for task in self.tasks:
                out[task] = F.interpolate(self.heads[task](encoder_output), img_size, mode="bilinear")

            output["output_for_tasks"] = out
            return output



            # output = self.backbone(images)
            # # aligned_feas_dict, out_feas_dict = self.backbone(x)
            # ## aligned_feas_dict: dict of {tea_name: list of [B, C_T, H, W]}
            # ## out_feas_dict: dict of {task: list of [B, C, H, W]}
            # feature_for_tasks = output["feature_for_tasks"]
            # for task in self.tasks:
            #     inter_in = self.heads[task](feature_for_tasks[task])
            #     # print(f"inter_in shape: {inter_in.shape}")
            #     # print(img_size)
            #     out[task] = F.interpolate(inter_in, img_size, mode="bilinear")
            # output["output_for_tasks"] = out
            # return output
