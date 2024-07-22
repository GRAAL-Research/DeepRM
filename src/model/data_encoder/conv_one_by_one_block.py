import torch
import torch.nn as nn

from src.model.data_encoder.data_encoder import DataEncoder


class ConvOneByOneBlock(DataEncoder):
    def __init__(self, n_features: int, n_instances_per_dataset_fed_to_deep_rm: int, n_filters: int,
                 is_target_provided: bool) -> None:
        super().__init__()
        self.output_dim = n_filters * n_instances_per_dataset_fed_to_deep_rm

        n_input_channels = n_features
        if is_target_provided:
            n_input_channels += 1

        self.conv_one_by_one = nn.Conv1d(n_input_channels, n_filters, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_one_by_one(x)

        return x.reshape(x.shape[0], -1)

    def get_output_dimension(self) -> int:
        return self.output_dim
