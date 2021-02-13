import os
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


class ColorizationDataset(Dataset):
    def __init__(self, size=128, num_images=-1, path=None, is_training=True):
        self.image_size = size
        self.is_training = is_training
        self.path = path

        self.image_paths = sorted(os.listdir(self.path))

        if num_images == -1:
            num_images = len(self.image_paths)

        self.image_file_name = np.random.choice(
            self.image_paths, size=num_images, replace=False
        )

        if self.is_training:
            self.transforms = T.Compose(
                [
                    T.Resize((self.image_size, self.image_size), Image.BICUBIC),
                    T.RandomVerticalFlip(0.5),
                    T.RandomHorizontalFlip(0.5),
                ]
            )
        else:
            self.image_file_name = sorted(self.image_file_name)
            self.transforms = T.Compose(
                [T.Resize((self.image_size, self.image_size), Image.BICUBIC),]
            )

    def __getitem__(self, idx):
        file_name = self.image_file_name[idx]
        img = Image.open(os.path.join(self.path, file_name)).convert("RGB")
        img = self.transforms(img)

        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b

        img_lab = T.ToTensor()(img_lab)

        L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.image_file_name)


def make_dataloaders(
    batch_size=32, n_workers=4, shuffle=True, **kwargs
):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)

    pin = True if torch.cuda.is_available() else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin,
        shuffle=shuffle,
    )
    return dataloader
