from src.utils.utils import TEST_ACCURACY, TRAIN_ACCURACY, VALID_ACCURACY, LINEAR_BOUND, \
    HYP_BOUND, KL_BOUND, MARCHAND_BOUND


def is_run_already_done(config: dict) -> bool:
    cnt_nw = 0
    new, keys = [], []
    for key in config:
        keys.append(key)
    keys.sort()
    for key in keys:
        new.append(str(config[key]).replace("\n", ""))
    try:
        with open("results/" + config["project_name"] + ".txt", "r") as tes:
            tess = [line.strip().split("\t") for line in tes]
        tes.close()
        for i in range(len(tess)):
            if tess[i][:len(new)] == new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open(config["project_name"] + ".txt", "a")
        for key in keys:
            file.write(key + "\t")
        file.write(TRAIN_ACCURACY + "\t")
        file.write(VALID_ACCURACY + "\t")
        file.write(TEST_ACCURACY + "\t")
        file.write(LINEAR_BOUND + "\t")
        file.write(HYP_BOUND + "\t")
        file.write(KL_BOUND + "\t")
        file.write(MARCHAND_BOUND + "\t")
        file.write("n_sigma" + "\t")
        file.write("n_Z" + "\n")
        file.close()

    return cnt_nw > 0


def save_run_in_a_text_file(config: dict, hist, best_epoch: int) -> None:
    """
    Writes in a .txt file the hyperparameters and results of a training of the BGN algorithm
        on a given dataset.
    Args:
        hist (dictionary): A dictionary that keep track of training metrics.
        best_epoch (int): best epoch.
    """
    keys = []
    for key in config:
        keys.append(key)
    keys.sort()
    file = open("results/" + config["project_name"] + ".txt", "a")
    for key in keys:
        file.write(str(config[key]).replace("\n", "") + "\t")
    file.write(str(hist[TRAIN_ACCURACY][best_epoch].item()) + "\t")
    file.write(str(hist[VALID_ACCURACY][best_epoch].item()) + "\t")
    file.write(str(hist[TEST_ACCURACY][best_epoch].item()) + "\t")
    file.write(str(hist[LINEAR_BOUND][best_epoch].item()) + "\t")
    file.write(str(hist[HYP_BOUND][best_epoch].item()) + "\t")
    file.write(str(hist[KL_BOUND][best_epoch]) + "\t")
    file.write(str(hist[MARCHAND_BOUND][best_epoch]) + "\t")
    file.write("\n")
    file.close()
