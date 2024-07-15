from typing import Callable

import torch

from src.model.utils.loss import l1, l2


def create_message_penalty_function(config: dict) -> Callable[[torch.tensor, float], torch.tensor]:
    if config["msg_penalty"].lower() == "l1":
        return l1

    if config["msg_penalty"].lower() == "l2":
        return l2

    raise NotImplementedError(f"The optimizer '{config['optimizer']}' is not supported.")
