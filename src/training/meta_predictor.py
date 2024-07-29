from torch import nn

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet


def create_meta_predictor(config: dict, predictor: Predictor) -> SimpleMetaNet:
    if config["meta_pred"].lower() == "simple_net":
        return SimpleMetaNet(config, predictor.n_params)

    raise NotImplementedError(f"The optimizer '{config['optimizer']}' is not supported.")
