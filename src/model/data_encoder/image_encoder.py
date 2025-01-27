import numpy as np
import torch
from transformers import ViTFeatureExtractor, ViTModel

from src.model.data_encoder.data_encoder import DataEncoder

VIT_EXPECTED_IMG_SIZE = 224
IMAGE_CHANNELS_MEANS = (0.5, 0.5, 0.5)
IMAGE_CHANNELS_STDS = (0.5, 0.5, 0.5)


class ImageEncoder(DataEncoder):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.target_size = config["target_size"]

        if config["device"] == "gpu":
            device = "cuda"
        elif config["device"] == "cpu":
            device = "cpu"
        else:
            raise NotImplementedError(f"{config['device']} is not supported")

        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                     do_resize=True, size=VIT_EXPECTED_IMG_SIZE,
                                                                     image_mean=IMAGE_CHANNELS_MEANS,
                                                                     image_std=IMAGE_CHANNELS_STDS)
        self.model = ViTModel.from_pretrained(model_name).to(device)
        self.device = device
        self.input_shape = config["input_shape"]
        self.n_features = config["n_features"]

    def forward(self, instances: torch.tensor) -> torch.tensor:
        batch_features = instances[:, :, :-self.target_size]

        n_channels = self.input_shape[0]
        img_height_or_width = int(np.sqrt(self.n_features / n_channels))
        meta_batch_size = len(batch_features)
        batch_size = len(batch_features[0])
        batch_features = batch_features.reshape(meta_batch_size * batch_size, n_channels, img_height_or_width,
                                                img_height_or_width)
        pytorch_tensor_type = "pt"

        if n_channels == 1:
            batch_features = batch_features.repeat(1, 3, 1, 1)

        model_inputs = self.feature_extractor(images=batch_features,
                                              return_tensors=pytorch_tensor_type)
        if n_channels == 1:
            model_outputs = self.model(**model_inputs.to("cuda"))

        return model_outputs.last_hidden_state[:, 0, :]

    def get_output_dimension(self) -> int:
        return self.model.config.hidden_size
