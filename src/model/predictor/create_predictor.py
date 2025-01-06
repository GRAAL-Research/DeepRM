from src.model.predictor.linear_classifier import LinearClassifier
from src.model.predictor.predictor import Predictor
from src.model.predictor.small_neural_network import FCNet, ConvNet


def create_predictor(config: dict) -> Predictor:
    if config["predictor"] == "LinearClassifier":
        return LinearClassifier(config)
    elif config["predictor"] == "FCNet":
        return FCNet(config)
    elif config["predictor"] == "ConvNet":
        return ConvNet(config)