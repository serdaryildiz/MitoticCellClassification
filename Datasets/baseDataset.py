import os.path
from typing import List

import numpy
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """
    Base Dataset Class
        takes dataset root!
        Dataset structure:
            - Dataset_ROOT
                - Class_1
                    - 0.jpg
                    ..
                - Class_2
                    - 0.jpg
                    ..
    """
    def __init__(self, dataset_root: str, transforms=None, class_names: List[str] = None):
        assert os.path.exists(dataset_root), f"Dataset not found! : {dataset_root}"

        self.images = []
        self.targets = []
        self.target_map = {}
        self.target_reverse_map = {}
        self.transforms = transforms

        if class_names is None:
            class_names = list(sorted(os.listdir(dataset_root)))
        # print("Class Names : ", class_names)

        # read all images and match with targets
        for class_id, class_name in enumerate(class_names):
            class_root = os.path.join(dataset_root, class_name)
            assert os.path.exists(class_root), f"class not found : {class_root}"

            self.target_map[class_name] = class_id
            self.target_reverse_map[class_id] = class_name

            for img_name in os.listdir(class_root):
                image_path = os.path.join(class_root, img_name)
                self.images.append(image_path)
                self.targets.append(class_id)
        return

    def __getitem__(self, idx):
        # get image
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        # get label
        label = self.targets[idx]
        label = torch.tensor(label)
        return idx, image, label

    def __len__(self):
        return len(self.targets)

    def get_name2id_map(self) -> dict:
        return self.target_map

    def get_id2name_map(self) -> dict:
        return self.target_reverse_map
