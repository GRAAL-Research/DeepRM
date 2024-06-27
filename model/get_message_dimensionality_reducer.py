from model.kme import KME


def get_message_dimensionality_reducer(task_dict):
    msg_dim_reducer = task_dict["msg_dim_reducer"]

    if msg_dim_reducer == "KME":
        return KME(task_dict["d"] + 1, task_dict["kme_dim"], task_dict["device"], task_dict["init"])

    raise NotImplementedError(f"'{msg_dim_reducer}' is not supported.")
