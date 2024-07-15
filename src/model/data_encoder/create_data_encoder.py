from src.model.data_encoder.data_encoder import DataEncoder
from src.model.data_encoder.concatenator import Concatenator
from src.model.data_encoder.conv_one_by_one_block import ConvOneByOneBlock
from src.model.data_encoder.fspool import FSPool
from src.model.data_encoder.kme import KME


def create_data_compressor_1(config: dict) -> DataEncoder:
    data_encoder_name = config["data_encoder_name"]

    if data_encoder_name.lower() == "kme":
        return KME(config, hidden_dims=config["kme_dim"])

    if data_encoder_name.lower() == "concatenator":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return Concatenator(config, n_instances_per_dataset_fed_to_deep_rm, is_target_provided=True)

    if data_encoder_name.lower() == "conv_one_by_one":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return ConvOneByOneBlock(config["n_features"], n_instances_per_dataset_fed_to_deep_rm, 
                                 config["conv_one_by_one_n_filters"], is_target_provided=True)

    if data_encoder_name.lower() == "fs_pool":
        return FSPool(config, config["fs_pool_dim"])

    raise NotImplementedError(f"'{data_encoder_name}' is not supported.")
