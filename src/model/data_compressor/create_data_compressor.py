from src.model.data_compressor.DataEncoder import DataEncoder
from src.model.data_compressor.concatenator import Concatenator
from src.model.data_compressor.conv_one_by_one_block import ConvOneByOneBlock
from src.model.data_compressor.fspool import FSPool
from src.model.data_compressor.kme import KME


def create_data_compressor_1(config: dict) -> DataEncoder:
    data_compressor_name = config["data_compressor_name"]

    if data_compressor_name.lower() == "kme":
        return KME(config, hidden_dims=config["kme_dim"])

    if data_compressor_name.lower() == "concatenator":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return Concatenator(config, n_instances_per_dataset_fed_to_deep_rm, is_target_provided=True)

    if data_compressor_name.lower() == "conv_one_by_one":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return ConvOneByOneBlock(config["n_features"], n_instances_per_dataset_fed_to_deep_rm, 
                                 config["conv_one_by_one_n_filters"], is_target_provided=True)

    if data_compressor_name.lower() == "fs_pool":
        return FSPool(config, config["fs_pool_dim"])

    raise NotImplementedError(f"'{data_compressor_name}' is not supported.")
