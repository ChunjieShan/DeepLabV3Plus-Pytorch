import torch
import os
import numpy as np
import random

from torch.utils.data.dataset import Dataset
from PIL import Image


class CUBSDataset(Dataset):
    def __init__(self, root, split="train", transform=None, train_split_ratio=0.8):
        super().__init__()
        self.root = root
        self.anno_path = os.path.join(self.root, "masks")
        self.images_path = os.path.join(self.root, "IMAGES")
        self.split_ratio = train_split_ratio
        self.transform = transform

        self.training_list, self.val_list = self._split_dataset()
        if split == "train":
            self.data_list = self.training_list
        else:
            self.data_list = self.val_list

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.data_list[idx])
        image_name = self.data_list[idx].split(".tiff")[0]
        img = Image.open(image_path).convert("RGB")
        target = Image.open(os.path.join(self.anno_path, image_name + ".png"))
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.data_list)

    def _split_dataset(self):
        images_list = os.listdir(self.images_path)
        random.shuffle(images_list)
        total_samples_num = len(images_list)
        training_num = int(total_samples_num * self.split_ratio)

        training_list = images_list[:training_num]
        val_list = images_list[training_num:]

        return training_list, val_list


if __name__ == '__main__':
    from utils import ext_transforms as et
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    data_root = "/mnt/h/Dataset/2.Carotid-Artery/DATASET_CUBS_tech"
    dataset = CUBSDataset(data_root, split="train", transform=train_transform)

    for _ in dataset:
        pass
