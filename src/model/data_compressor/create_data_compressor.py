from src.model.data_compressor.kme import KME


def create_data_compressor_1(config: dict):
    data_compressor_name = config["data_compressor_name"]

    if data_compressor_name == "KME":
        return KME(config["n_features"] + 1, config["data_compressor_dim"], config["device"], config["init_scheme"],
                   config["has_skip_connection"], config["has_batch_norm"])

    raise NotImplementedError(f"'{data_compressor_name}' is not supported.")
