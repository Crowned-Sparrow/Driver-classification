import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class RAFDBDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        self.class_names = sorted(os.listdir(root_dir))

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.class_names)
        }

        for cls_name in self.class_names:

            cls_dir = os.path.join(root_dir, cls_name)

            for file_name in os.listdir(cls_dir):

                if file_name.endswith((
                    ".jpg",
                    ".png",
                    ".jpeg"
                )):

                    img_path = os.path.join(
                        cls_dir,
                        file_name
                    )

                    label = self.class_to_idx[cls_name]

                    self.samples.append(
                        (img_path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
