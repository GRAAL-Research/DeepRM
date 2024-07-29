from src.model.predictor.linear_classifier import LinearClassifier
from src.model.predictor.predictor import Predictor
from src.model.predictor.small_neural_network import SmallNeuralNetwork


def create_predictor(config: dict) -> Predictor:
    if len(config["pred_hidden_sizes"]) == 0:
        return LinearClassifier(config)

    return SmallNeuralNetwork(config)
