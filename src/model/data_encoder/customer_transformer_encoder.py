import torch.nn
from torch import nn

from src.model.data_encoder.data_encoder import DataEncoder


class CustomTransformerEncoder(DataEncoder):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config["tfm_output_dim"] * config["n_instances_per_dataset"] // 2

        self.input_projection_layer = nn.Linear(config["n_features"] + config["target_size"], config["tfm_input_dim"])
        encoder_layer = nn.TransformerEncoderLayer(config["tfm_input_dim"], config["tfm_n_heads"],
                                                   config["tfm_mlp_dim"], dropout=config["tfm_drop_out"],
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config["tfm_n_encoders"])
        self.output_projection_layer = nn.Linear(config["tfm_input_dim"], config["tfm_output_dim"])

        if config["device"] == "gpu":
            self.cuda()

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        outputs = self.input_projection_layer(instances)
        outputs = self.transformer_encoder(outputs)
        outputs = self.output_projection_layer(outputs)

        return outputs.reshape(len(outputs), -1)

    def get_output_dimension(self) -> int:
        return self.output_dim
