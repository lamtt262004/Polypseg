import itertools
import random

import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


train_transform = A.Compose([
    A.Resize(256, 256),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PolypDS(Dataset):
    def __init__(self, data_path, split, transform=None):
        super().__init__()
        data = np.load(data_path)
        self.images = data[f"{split}_img"]
        self.masks = data[f"{split}_msk"].squeeze(-1)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform is not None:
            out = self.transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = transforms.ToTensor()(img)
        msk = transforms.ToTensor()(np.expand_dims(msk, axis=-1))
        return img, msk

    def __len__(self):
        return len(self.images)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__()
        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.secondary_indices = self.indices[primary_count:]
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for primary_batch, secondary_batch in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
