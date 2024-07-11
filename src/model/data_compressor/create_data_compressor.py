from src.model.data_compressor.DataEncoder import DataEncoder
from src.model.data_compressor.concatenator import Concatenator
from src.model.data_compressor.conv_one_by_one_block import ConvOneByOneBlock
from src.model.data_compressor.kme import KME


def create_data_compressor_1(config: dict) -> DataEncoder:
    data_compressor_name = config["data_compressor_name"]

    if data_compressor_name.lower() == "kme":
        return KME(config["n_features"] + 1, config["data_compressor_dim"], config["device"], config["init_scheme"],
                   config["has_skip_connection"], config["has_batch_norm"], config["task"])

    if data_compressor_name.lower() == "concatenator":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return Concatenator(config, n_instances_per_dataset_fed_to_deep_rm, is_target_provided=True)

    if data_compressor_name.lower() == "conv_one_by_one":
        n_instances_per_dataset_fed_to_deep_rm = config["n_instances_per_dataset"] // 2
        return ConvOneByOneBlock(config["n_features"], n_instances_per_dataset_fed_to_deep_rm, 
                                 config["n_filters"], is_target_provided=True)

    raise NotImplementedError(f"'{data_compressor_name}' is not supported.")
