import torch
import torchvision.transforms as transforms
from typing import Dict, Any, List


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


class AdjustContrast:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, tensor):
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        return (tensor - mean) * self.factor + mean


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    image_size = config.get('image_size', [224, 224])
    normalize = config.get('normalize', True)
    mean = config.get('mean', [0.485, 0.456, 0.406])
    std = config.get('std', [0.229, 0.224, 0.225])

    transform_list = [
        transforms.RandomResizedCrop(image_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_evaluation_transforms(
    noise_std: float = 0.0,
    contrast_factor: float = 1.0,
    subsampling_step: int = 1,
    image_size: List[int] = [224, 224],
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    transform_list = [
        transforms.Resize(int(image_size[0] * 1.143)),
        transforms.CenterCrop(image_size[0]),
        transforms.ToTensor(),
    ]

    if contrast_factor != 1.0:
        transform_list.append(AdjustContrast(factor=contrast_factor))

    if noise_std > 0:
        transform_list.append(AddGaussianNoise(std=noise_std))

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    image_size = config.get('image_size', [224, 224])
    normalize = config.get('normalize', True)
    mean = config.get('mean', [0.485, 0.456, 0.406])
    std = config.get('std', [0.229, 0.224, 0.225])

    transform_list = [
        transforms.Resize(int(image_size[0] * 1.143)),
        transforms.CenterCrop(image_size[0]),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


EVAL_TRANSFORMS = {
    'original': {
        'noise_std': 0.0,
        'contrast_factor': 1.0,
    },
    'gaussian_noise_medium': {
        'noise_std': 0.1,
        'contrast_factor': 1.0,
    },
    'gaussian_noise_high': {
        'noise_std': 0.2,
        'contrast_factor': 1.0,
    },
    'contrast': {
        'noise_std': 0.0,
        'contrast_factor': 1.5,
    },
    'noise_contrast': {
        'noise_std': 0.1,
        'contrast_factor': 1.5,
    },
}


def build_transforms(transform_name: str, image_size: List[int] = [64, 64]):
    train = get_named_evaluation_transform(transform_name, image_size=image_size)
    val = get_named_evaluation_transform("original", image_size=image_size)
    return train, val


def get_named_evaluation_transform(
    transform_name: str,
    image_size: List[int] = [224, 224],
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    if transform_name not in EVAL_TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(EVAL_TRANSFORMS.keys())}")

    params = EVAL_TRANSFORMS[transform_name]
    return get_evaluation_transforms(
        noise_std=params['noise_std'],
        contrast_factor=params['contrast_factor'],
        subsampling_step=1,
        image_size=image_size,
        normalize=normalize,
        mean=mean,
        std=std
    )
