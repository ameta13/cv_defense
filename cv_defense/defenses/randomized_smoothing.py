import torch

from cv_defense.defenses.base_transform import BaseTransform


class RandomizedSmoothing(BaseTransform):
    """
    :param mean: Mean value for the Gaussian noise.
    :param stddev: Standard deviation for the Gaussian noise.
    :param clip_pixels: Need to clip pixel values to range [0, 255]
    """
    mean: float = 0.

    def __init__(self, stddev: float = 5, clip_pixels: bool = False):
        assert stddev >= 0., f'normal expects std >= 0.0, but found std {stddev}'
        self.stddev = stddev
        self.clip_pixels = clip_pixels


    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: Input images as a PyTorch Tensor of shape (N, C, H, W) where
            - N is the number of images
            - C is the number of channels
            - H is height
            - W is width
        :return: Smoothed images with Gaussian noise added.
        """
        noises = torch.normal(self.mean, self.stddev, size=images.shape)
        smoothed_images = (images + noises).to(dtype=torch.uint8)

        if self.clip_pixels:
            # clip pixel values to range [0, 255]
            maximum_mask = torch.full(size=images.shape, fill_value=255, dtype=torch.uint8)
            smoothed_images = torch.minimum(smoothed_images, maximum_mask)

            minimum_mask = torch.zeros(size=images.shape, dtype=torch.uint8)
            smoothed_images = torch.maximum(smoothed_images, minimum_mask)

        return smoothed_images
