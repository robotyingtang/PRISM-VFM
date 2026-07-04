from torch.utils.data import DataLoader

from .utils.custom_collate import collate_mil
from utils import global_print
from torchvision.datasets import ImageFolder
import os
import torch
import numpy as np


def get_dataset(dataname, train, tasks, transform, dataidxs=None):
    """
    Get the dataset
    """
    if train:
        global_print("Get training dataset for %s" % (dataname))
    else:
        global_print("Get validation dataset for %s" % (dataname))

    if dataname == "pascalcontext":
        from .pascal_context import PASCALContext

        data_root = os.environ.get("DATA_ROOT", "data")
        database = PASCALContext(
            os.path.join(data_root, "PASCALContext"), train=train, transform=transform, tasks=tasks, dataidxs=dataidxs
        )
    elif dataname == "nyud":
        from .nyud import NYUD

        data_root = os.environ.get("DATA_ROOT", "data")
        database = NYUD(
            os.path.join(data_root, "NYUDv2"), train=train, transform=transform, tasks=tasks, dataidxs=dataidxs
        )
    elif dataname == "imagenet":
        split = "train"
        imagenet_root = os.environ.get("IMAGENET_ROOT", "data/imagenet-1k")
        database = ImageFolder(os.path.join(imagenet_root, split), transform=transform, target_transform=torch.tensor)
    else:
        raise NotImplementedError("'dataname': choose among 'pascalcontext', 'nyud', and 'imagenet'.")

    return database

# global N_TEACHERS #TODO : make this configurable
N_TEACHERS = 3
def collate_fn_with_random_teacher_assignment(batch_list):
    # batch_list 是一个列表，每个元素是 (image_tensor, label_tensor)
    images = torch.stack([item[0] for item in batch_list])
    labels = torch.stack([item[1] for item in batch_list]) # 假设标签也是tensor
    batch_size = len(batch_list) # 例如 129

    # 生成教师ID
    # 创建一个教师ID列表，确保每个教师ID大致出现 batch_size / N_TEACHERS 次
    ids_per_teacher = batch_size // N_TEACHERS
    remainder = batch_size % N_TEACHERS

    teacher_ids_list = []
    for i in range(N_TEACHERS):
        count = ids_per_teacher + (1 if i < remainder else 0)
        teacher_ids_list.extend([i] * count)

    # 打乱教师ID列表，然后分配给批次中的样本
    np.random.shuffle(teacher_ids_list)
    vfm_teacher_ids_tensor = torch.tensor(teacher_ids_list, dtype=torch.long)

    return {
        "image": images,
        "label": labels,
        "vfm_teacher_id": vfm_teacher_ids_tensor,
    }

def collate_fn_with_random_teacher_assignment_for_pascal(batch_list):
    batch_size = len(batch_list)
    images = torch.stack([item['image'] for item in batch_list])
    batch_labels={}
    if 'semseg' in batch_list[0]:
        batch_labels['semseg'] = torch.stack([item['semseg'] for item in batch_list])
    if 'human_parts' in batch_list[0]:
        batch_labels['human_parts'] = torch.stack([item['human_parts'] for item in batch_list])
    if 'normals' in batch_list[0]:
        batch_labels['normals'] = torch.stack([item['normals'] for item in batch_list])
    if 'edge' in batch_list[0]:
        batch_labels['edge'] = torch.stack([item['edge'] for item in batch_list])
    if 'sal' in batch_list[0]:
        batch_labels['sal'] = torch.stack([item['sal'] for item in batch_list])
    if 'depth' in batch_list[0]:
        batch_labels['depth'] = torch.stack([item['depth'] for item in batch_list])

    # batch_labels['semseg'] = torch.stack([item['semseg'] for item in batch_list])
    # batch_labels['human_parts'] = torch.stack([item['human_parts'] for item in batch_list])
    # batch_labels['normals'] = torch.stack([item['normals'] for item in batch_list])
    # batch_labels['edge'] = torch.stack([item['edge'] for item in batch_list])
    # batch_labels['sal'] = torch.stack([item['sal'] for item in batch_list])
    meta={}
    meta["file_name"] = [item['meta']["file_name"] for item in batch_list]  # 保留为字符串列表
    meta["size"] = [item['meta']["size"] for item in batch_list]
    ids_per_teacher = batch_size // N_TEACHERS
    remainder = batch_size % N_TEACHERS
    teacher_ids_list = []

    teacher_order = np.random.permutation(N_TEACHERS)
    for i in range(N_TEACHERS):
        count = ids_per_teacher + (1 if i < remainder else 0)
        teacher_id = teacher_order[i]
        teacher_ids_list.extend([teacher_id] * count)

    # 打乱教师ID列表，然后分配给批次中的样本

    np.random.shuffle(teacher_ids_list)
    vfm_teacher_ids_tensor = torch.tensor(teacher_ids_list, dtype=torch.long)
    return {"image": images,
            "label": batch_labels,
            "vfm_teacher_id": vfm_teacher_ids_tensor,
            "meta": meta
             }
def get_dataloader(train, configs, dataset, sampler=None):
    """
    Get the dataloader from dataset
    """
    if train:
        if dataset.__class__.__name__ in ["PASCALContext", "NYUD"]:
            collate_fn_function = collate_fn_with_random_teacher_assignment_for_pascal
        else:
            collate_fn_function = collate_fn_with_random_teacher_assignment

        dataloader = DataLoader(
            dataset,
            batch_size=configs["tr_batch"],
            drop_last=True,
            num_workers=configs["nworkers"],
            collate_fn=collate_fn_function,
            pin_memory=True,
            sampler=sampler,
        )


    else:
        if dataset.__class__.__name__ in ["PASCALContext", "NYUD"]:
            collate_fn_function = collate_fn_with_random_teacher_assignment_for_pascal
        else:
            collate_fn_function = collate_fn_with_random_teacher_assignment
        dataloader = DataLoader(
            dataset,
            batch_size=configs["val_batch"],
            shuffle=False,
            drop_last=False,
            num_workers=configs["nworkers"],
            collate_fn=collate_fn_function,
            pin_memory=True,
        )
    return dataloader
