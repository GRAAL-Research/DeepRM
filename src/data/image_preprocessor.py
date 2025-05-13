import torch
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel

VIT_EXPECTED_IMG_SIZE = 224
IMAGE_CHANNELS_MEANS = (0.5, 0.5, 0.5)
IMAGE_CHANNELS_STDS = (0.5, 0.5, 0.5)
OUTPUT_FEATURE_SIZE = 768
MAX_BATCH_SIZE = 512
EXPECTED_INPUT_DIMENSION = 4


class ImagePreprocessor(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.target_size = config["target_size"]

        if config["device"] == "gpu":
            self.device = "cuda"
        elif config["device"] == "cpu":
            self.device = "cpu"
        else:
            raise NotImplementedError(f"{config['device']} is not supported")

        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                     do_resize=True, size=VIT_EXPECTED_IMG_SIZE,
                                                                     image_mean=IMAGE_CHANNELS_MEANS,
                                                                     image_std=IMAGE_CHANNELS_STDS)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.n_features = config["n_features"]

    def forward(self, features: torch.tensor) -> torch.tensor:
        error_msg = "It is expected a batch of images : batch x channel x height x width."
        assert len(features.shape) == EXPECTED_INPUT_DIMENSION, error_msg

        n_channels = features.shape[1]
        if n_channels == 1:
            features = features.repeat(1, 3, 1, 1)

        n_batches = features.shape[0]
        if n_batches > MAX_BATCH_SIZE:
            return self._process_too_much_batches(features)

        return self._process_batches(features)

    def _process_too_much_batches(self, features: torch.tensor) -> torch.tensor:
        outputs = []
        n_batches = features.shape[0]

        for batch_idx in range(0, n_batches, MAX_BATCH_SIZE):
            batch_features = features[batch_idx:batch_idx + MAX_BATCH_SIZE]
            outputs.append(self._process_batches(batch_features))

        return torch.cat(outputs, dim=0)

    def _process_batches(self, batch_features: torch.tensor) -> torch.tensor:
        pytorch_tensor_type = "pt"
        with torch.no_grad():
            model_inputs = self.feature_extractor(images=batch_features,
                                                  return_tensors=pytorch_tensor_type)
            model_outputs = self.model(**model_inputs.to(self.device))

        return model_outputs.pooler_output

    @staticmethod
    def get_output_dimension() -> int:
        return OUTPUT_FEATURE_SIZE
