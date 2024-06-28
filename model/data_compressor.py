from model.kme import KME


def create_data_compressor_1(config: dict):
    msg_dim_reducer = config["data_compressor_name"]

    if msg_dim_reducer == "KME":
        return KME(config["d"] + 1, config["data_compressor_dim"], config["device"], config["init"])

    raise NotImplementedError(f"'{msg_dim_reducer}' is not supported.")
