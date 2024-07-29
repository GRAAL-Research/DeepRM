import torch

from src.model.data_encoder.data_encoder import DataEncoder


class Concatenator(DataEncoder):
    def __init__(self, config: dict, n_instances_per_dataset_fed_to_deep_rm: int, is_target_provided: bool) -> None:
        super().__init__()
        self.device = config["device"]

        instance_length = config["n_features"]
        if is_target_provided:
            instance_length += config["target_size"]
        self.output_dim = n_instances_per_dataset_fed_to_deep_rm * instance_length

    def forward(self, datasets: torch.Tensor) -> torch.Tensor:
        n_instances_per_dataset_dim_idx = 1
        shuffled_datasets_by_instance_per_dataset = self.shuffle(datasets, n_instances_per_dataset_dim_idx)

        return shuffled_datasets_by_instance_per_dataset.flatten(n_instances_per_dataset_dim_idx)

    def shuffle(self, instances: torch.Tensor, dim: int = 0) -> torch.Tensor:
        random_indices = torch.randperm(instances.shape[dim])
        if self.device == "gpu":
            random_indices = random_indices.cuda()

        return instances.index_select(dim, random_indices)

    def get_output_dimension(self) -> int:
        return self.output_dim
