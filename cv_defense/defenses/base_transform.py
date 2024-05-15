from abc import ABC, abstractmethod

import torch


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: Input images as a PyTorch Tensor of shape (N, C, H, W) where
            - N is the number of images
            - C is the number of channels
            - H is height
            - W is width
        :return: images with jpg compression
        """
        return images
