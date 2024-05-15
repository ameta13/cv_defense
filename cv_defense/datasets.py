from enum import StrEnum
import os

import pandas as pd
from PIL import Image
import torchvision
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        root,
        train: bool = True,
        transform = None,
        path_to_train: str = 'train.csv',
        path_to_test: str = 'test.csv',
        column_path: str = 'path',
        column_label: str = 'label',
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.column_path = column_path
        self.column_label = column_label

        if self.train:
            file_path = os.path.join(self.root, path_to_train)
        else:
            file_path = os.path.join(self.root, path_to_test)

        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.data.iloc[index][self.column_path])
        img = Image.open(img_path)
        label = self.data.iloc[index][self.column_label]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_all_image_paths() -> list[str]:
        return list(self.data[self.column_path])


def get_dataset(dataset_name: str, train: bool = True, transform = None, download: bool = True):
    allowed_datasets = {'MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'CIFAR10', 'ImageNet'}
    assert dataset_name in allowed_datasets, f'Unexpected {dataset_name=}. Allowed: {allowed_datasets}'

    if dataset_name in {'MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'CIFAR10'}:
        dataset = getattr(torchvision.datasets, dataset_name)(
            root="./data",
            train=train,
            transform=transform,
            download=download,
        )
    elif dataset == 'ImageNet':
        dataset = torchvision.datasets.ImageNet(
            root="./data",
            split='train' if train else 'val',
            transform=transform,
            download=download,
        )
    else:
        raise ValueError(f'Unexpected {dataset_name=}')
