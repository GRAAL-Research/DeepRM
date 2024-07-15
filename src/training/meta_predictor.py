from torch import nn

from src.model.simple_meta_net import SimpleMetaNet


def create_meta_predictor(config: dict, predictor: nn.Module) -> nn.Module:
    if config["meta_pred"].lower() == "simple_net":
        return SimpleMetaNet(config, predictor.n_param)

    raise NotImplementedError(f"The optimizer '{config['optimizer']}' is not supported.")
