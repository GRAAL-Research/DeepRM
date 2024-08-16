import numpy as np
import wandb
from torch.utils.data import DataLoader

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.result.decision_boundaries import show_decision_boundaries


def compute_medias(config: dict, meta_predictor: SimpleMetaNet, test_loader: DataLoader, predictor: Predictor,
                   train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray, datasets: np.ndarray,
                   classes_name) -> None:
    if config["n_features"] == 2 and config["is_using_wandb"]:
        show_decision_boundaries(meta_predictor, config["dataset"], test_loader, predictor, wandb, config["device"])

    #if config["dataset"] in ["mnist", "cifar100_binary"]:
    #    idx = [train_idx, valid_idx, test_idx]
    #    show_performance_matrix(meta_predictor, predictor, config["dataset"], datasets, classes_name, idx,
    #                            config["n_dataset"], config["is_using_wandb"], wandb, config["batch_size"],
    #                            config["are_test_classes_shared_with_train"], config["device"])
