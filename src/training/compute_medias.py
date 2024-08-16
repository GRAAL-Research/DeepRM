import numpy as np
import wandb
from torch.utils.data import DataLoader

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.result.decision_boundaries import show_decision_boundaries


def compute_medias(config: dict, meta_predictor: SimpleMetaNet, test_loader: DataLoader, predictor: Predictor) -> None:
    if config["n_features"] == 2 and config["is_using_wandb"]:
        show_decision_boundaries(meta_predictor, config["dataset"], test_loader, predictor, wandb, config["device"])
        
