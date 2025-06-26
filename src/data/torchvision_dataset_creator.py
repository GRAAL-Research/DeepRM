from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.data.cache import create_cache_path, store_datasets_in_cache, create_cache_base_path
from src.data.dataset.torchvision_dataset.torchvision_dataset import TorchvisionDataset


class TorchvisionDatasetCreator:
    def __init__(self, torchvision_dataset: TorchvisionDataset):
        self.torchvision_dataset = torchvision_dataset

    def create_meta_dataset(self, config: dict) -> np.ndarray:
        """
        Load the data and store in cache.
        """
        data_base_path = self.torchvision_dataset.get_data_base_path()
        expected_datasets_cache_path = create_cache_path(config, create_cache_base_path(data_base_path))

        if expected_datasets_cache_path.exists():
            return np.load(expected_datasets_cache_path)

        train, test = self.create_train_and_test_datasets(config, data_base_path)
        datasets = self.torchvision_dataset.create_meta_datasets(config, train, test)
        store_datasets_in_cache(config, create_cache_base_path(data_base_path), datasets)

        return datasets

    def create_train_and_test_datasets(self, config: dict, dataset_base_path: Path) -> tuple[Tensor, Tensor]:
        train_features, train_targets = self.torchvision_dataset.download(dataset_base_path, is_train_data=True)
        test_features, test_targets = self.torchvision_dataset.download(dataset_base_path, is_train_data=False)
        train_set = self.preprocess_dataset(config, train_features, train_targets)
        test_set = self.preprocess_dataset(config, test_features, test_targets)

        return train_set, test_set

    @classmethod
    def preprocess_dataset(cls, config: dict, features: np.ndarray, targets: torch.Tensor) -> torch.Tensor:
        n_instances = len(targets)
        reshaped_targets = torch.tensor(targets).unsqueeze(-1) if type(targets) == list else targets.unsqueeze(-1)

        features = cls.apply_image_transforms_to_features(config, features)
        reshaped_features = features.reshape((n_instances, config["n_features"]))
        return torch.hstack((reshaped_features, reshaped_targets))

    @staticmethod
    def apply_image_transforms_to_features(config: dict, features: np.ndarray) -> torch.tensor:
        has_color_channels = len(features.shape) == 4

        if not has_color_channels:
            n_channels = 1
            features = np.expand_dims(features, -1)
        else:
            n_channels = features.shape[-1]

        square_root_of_n_features = np.sqrt(config["n_features"] // n_channels)
        is_a_perfect_square = 0 <= config["n_features"] // n_channels == int(square_root_of_n_features) ** 2
        error_msg = f"The number of features per channel ({config['n_features']}/{n_channels}) must be a perfect square (e.g. 784 = 1x28x28)."
        assert is_a_perfect_square, error_msg

        new_img_size = (int(square_root_of_n_features), int(square_root_of_n_features))

        is_new_size_identical_to_actual_size = new_img_size == (features.shape[-2], features.shape[-1])
        if is_new_size_identical_to_actual_size:
            return features

        transform = transforms.Compose(
            [to_pil_image, transforms.ToTensor(), transforms.Resize(new_img_size),
             transforms.Normalize((0.5,), (0.5,))])

        transformed_features = []
        for img in tqdm(features, desc="Applying transforms"):
            img = transform(img)
            transformed_features.append(img)

        return torch.stack(transformed_features).squeeze(1)
