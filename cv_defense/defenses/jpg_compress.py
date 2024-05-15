import io

from PIL import Image
import torch
from torchvision.transforms.v2 import ToPILImage, PILToTensor

from cv_defense.defenses.base_transform import BaseTransform


class JpgCompression(BaseTransform):
    def __init__(self, quality: int = 75):
        assert 1 <= quality <= 95, f'quality expected value between 1 and 95, got {quality=}'  # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving
        self.quality = quality

    def _compress_to_jpg(self, image: torch.Tensor) -> torch.Tensor:
        pil_image = ToPILImage()(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='jpeg', quality=self.quality)
        compressed_pil_image = Image.open(buffer)
        compressed_image = PILToTensor()(compressed_pil_image)
        return compressed_image

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: Input images as a PyTorch Tensor of shape (N, C, H, W) where
            - N is the number of images
            - C is the number of channels
            - H is height
            - W is width
        :return: images with jpg compression
        """
        assert images.ndim == 4, f'Expected ndim=4, got {images.ndim=}'
        assert images.shape[1] == 3, f'Expected 3 channels, got {images.shape[1]=}. {images.shape=}'
        return torch.vmap(self._compress_to_jpg, in_dims=0, out_dims=0)(images)
