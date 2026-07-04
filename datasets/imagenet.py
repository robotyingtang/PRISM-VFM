import tarfile
import os
import numpy as np
from PIL import Image
from torch.utils import data

from .utils.mypath import MyPath
from utils import global_print


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Sampler
import random



class ImageTarDataset(data.Dataset):
    def __init__(self, root=MyPath.db_root_dir("imagenet"), train=True, transform=None):
        """
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        """
        if train:
            tar_file = os.path.join(root, "train.tar")
        else:
            tar_file = os.path.join(root, "val.tar")
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, "r:") as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count("/") != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        global_print(
            "Loaded the dataset from {}. It contains {} samples.".format(
                tar_file, self.num
            )
        )
        self.transform = transform

    def _get_category_from_filename(self, filename):
        begin = filename.find("/")
        begin += 1
        end = filename.find("/", begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, "r:")

        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert("RGB")
        image = self.transform(image)

        return image


class ImageDataset(data.Dataset):
    def __init__(self, root=MyPath.db_root_dir("imagenet"), train=True, transform=None):
        """
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        """
        if train:
            tar_file = os.path.join(root, "train.tar")
        else:
            tar_file = os.path.join(root, "val.tar")
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, "r:") as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count("/") != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        global_print(
            "Loaded the dataset from {}. It contains {} samples.".format(
                tar_file, self.num
            )
        )
        self.transform = transform

    def _get_category_from_filename(self, filename):
        begin = filename.find("/")
        begin += 1
        end = filename.find("/", begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, "r:")

        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert("RGB")
        image = self.transform(image)

        return image



class ImageFolderWithTeacherID(Dataset):
    def __init__(self, root, transform=None, target_transform=None, num_vfm_teachers=3):
        self.image_folder = ImageFolder(root, transform=transform, target_transform=target_transform)
        self.num_vfm_teachers = num_vfm_teachers
        self.samples = self.image_folder.samples
        self.targets = self.image_folder.targets # 原始ImageFolder的类别标签

        # 为每个样本分配一个 vfm_teacher_id
        # 这里简单地轮询分配，你可以根据需求实现更复杂的分配逻辑
        self.vfm_teacher_ids_for_samples = [i % self.num_vfm_teachers for i in range(len(self.samples))]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.image_folder.loader(path)
        if self.image_folder.transform is not None:
            sample = self.image_folder.transform(sample)
        if self.image_folder.target_transform is not None:
            target = self.image_folder.target_transform(target)

        vfm_teacher_id = self.vfm_teacher_ids_for_samples[index]

        return {
            "image": sample,
            "label": target, # 原始标签
            "vfm_teacher_id": torch.tensor(vfm_teacher_id, dtype=torch.long) # VFM教师ID
        }

    def __len__(self):
        return len(self.samples)



class TeacherPartitionSampler(Sampler):
    def __init__(self, dataset: ImageFolderWithTeacherID, batch_per_teacher: int, drop_last: bool = True, shuffle: bool = True, ddp_rank=0, ddp_world_size=1):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_vfm_teachers = dataset.num_vfm_teachers
        self.batch_per_teacher = batch_per_teacher
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # 将数据集按 vfm_teacher_id 分组
        self.indices_by_teacher = [[] for _ in range(self.num_vfm_teachers)]
        for i, teacher_id in enumerate(self.dataset.vfm_teacher_ids_for_samples):
            self.indices_by_teacher[teacher_id].append(i)

        # 计算每个教师在当前DDP rank上应该处理的样本数量和批次数
        self.num_batches_per_teacher_per_rank = [
            len(indices) // (self.batch_per_teacher * self.ddp_world_size)
            for indices in self.indices_by_teacher
        ]
        # 取最小的批次数，以确保所有教师都能提供足够的批次，或者根据具体需求调整
        self.num_batches_total_per_rank = min(self.num_batches_per_teacher_per_rank)
        if self.num_batches_total_per_rank == 0 and not self.drop_last:
             # 如果数据太少，至少保证一个批次（如果drop_last=False）
             # 但这可能会导致某些教师的批次不足batch_per_teacher
             self.num_batches_total_per_rank = 1 if any(len(indices) >= self.batch_per_teacher for indices in self.indices_by_teacher) else 0


        self.total_size_per_rank = self.num_batches_total_per_rank * self.num_vfm_teachers * self.batch_per_teacher

        if self.total_size_per_rank == 0 and len(dataset) > 0 :
            print(f"Warning: Not enough data for rank {self.ddp_rank} to form even one full batch per teacher. "
                  f"Dataset size: {len(dataset)}, batch_per_teacher: {self.batch_per_teacher}, num_teachers: {self.num_vfm_teachers}, ddp_world_size: {self.ddp_world_size}")


    def __iter__(self):
        # 为每个教师的索引列表进行shuffle (如果需要)
        if self.shuffle:
            for indices in self.indices_by_teacher:
                random.shuffle(indices)

        # 为当前DDP rank生成批次索引
        # 每个rank负责一部分批次
        batch_indices = []
        for batch_num in range(self.num_batches_total_per_rank):
            current_batch_group_indices = []
            for teacher_id in range(self.num_vfm_teachers):
                start_idx_for_teacher_batch = (batch_num * self.ddp_world_size + self.ddp_rank) * self.batch_per_teacher
                end_idx_for_teacher_batch = start_idx_for_teacher_batch + self.batch_per_teacher

                teacher_specific_indices = self.indices_by_teacher[teacher_id]

                if end_idx_for_teacher_batch <= len(teacher_specific_indices):
                    current_batch_group_indices.extend(teacher_specific_indices[start_idx_for_teacher_batch:end_idx_for_teacher_batch])
                elif not self.drop_last and start_idx_for_teacher_batch < len(teacher_specific_indices):
                    # 如果不drop_last且还有剩余数据，则取剩余部分
                    current_batch_group_indices.extend(teacher_specific_indices[start_idx_for_teacher_batch:])
                # else: 如果drop_last或者数据不足，这个教师在这个批次可能不提供数据或提供不足的数据
                # 这个逻辑需要根据具体需求细化，如何处理数据不足的情况

            if len(current_batch_group_indices) == self.num_vfm_teachers * self.batch_per_teacher or \
               (not self.drop_last and len(current_batch_group_indices) > 0):
                batch_indices.extend(current_batch_group_indices)
            elif self.drop_last and len(current_batch_group_indices) > 0 and len(current_batch_group_indices) < self.num_vfm_teachers * self.batch_per_teacher:
                # 如果启用了drop_last，并且当前批次组不完整，则跳过（这可能导致某些批次迭代为空）
                # print(f"Rank {self.ddp_rank} skipping incomplete batch group due to drop_last. Collected {len(current_batch_group_indices)} indices.")
                pass


        # 如果shuffle整个批次流 (可选)
        # if self.shuffle:
        #     # 注意：这里打乱的是所有教师拼接后的索引流，而不是每个教师内部的批次
        #     # 这可能不是期望的行为，因为我们希望每个教师的43个样本是连续的
        #     pass

        return iter(batch_indices)

    def __len__(self):
        return self.total_size_per_rank // (self.num_vfm_teachers * self.batch_per_teacher) # 返回的是“元批次”的数量
        # 或者更准确地说是生成的索引总数
        # return self.total_size_per_rank
