def is_run_already_done(config: dict) -> bool:
    cnt_nw = 0
    new, keys = [], []
    for key in config:
        keys.append(key)
    keys.sort()
    for key in keys:
        new.append(str(config[key]))
    try:
        with open("results/" + config["project_name"] + ".txt", "r") as tes:
            tess = [line.strip().split("\t") for line in tes]
        tes.close()
        for i in range(len(tess)):
            if tess[i][:len(new)] == new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open("results/" + config["project_name"] + ".txt", "a")
        for key in keys:
            file.write(key + "\t")
        file.write("train_acc" + "\t")
        file.write("valid_acc" + "\t")
        file.write("test_acc" + "\t")
        file.write("bound_lin" + "\t")
        file.write("bound_hyp" + "\t")
        file.write("bound_kl" + "\t")
        file.write("bound_mrch" + "\t")
        file.write("n_sigma" + "\t")
        file.write("n_Z" + "\n")
        file.close()

    return cnt_nw > 0


def save_run_in_a_text_file(config: dict, hist, best_epoch):
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
        file.write(str(config[key]) + "\t")
    file.write(str(hist["train_acc"][best_epoch].item()) + "\t")
    file.write(str(hist["valid_acc"][best_epoch].item()) + "\t")
    file.write(str(hist["test_acc"][best_epoch].item()) + "\t")
    file.write(str(hist["bound_lin"][best_epoch].item()) + "\t")
    file.write(str(hist["bound_hyp"][best_epoch].item()) + "\t")
    file.write(str(hist["bound_kl"][best_epoch]) + "\t")
    file.write(str(hist["bound_mrch"][best_epoch]) + "\t")
    file.write("\n")
    file.close()
