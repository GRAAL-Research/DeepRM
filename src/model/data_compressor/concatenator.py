import torch


class Concatenator:
    def __init__(self, config: dict) -> None:
        self.device = config["device"]

    def forward(self, datasets: torch.Tensor) -> torch.Tensor:
        n_instances_per_dataset_dim_idx = 1
        shuffled_datasets_by_instance_per_dataset = self.shuffle(datasets, n_instances_per_dataset_dim_idx)
        target_idx = -1
        x = shuffled_datasets_by_instance_per_dataset[:, :, :target_idx]

        return x.flatten(n_instances_per_dataset_dim_idx)

    def shuffle(self, instances: torch.Tensor, dim: int = 0) -> torch.Tensor:
        random_indices = torch.randperm(instances.shape[dim])
        if self.device == "gpu":
            random_indices = random_indices.cuda()

        return instances.index_select(dim, random_indices)
