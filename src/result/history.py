def update_hist(hist, values):
    """
    Adds values to the hist. dictionary to keep track of the losses and accuracies for each epoch.
    Args:
        hist (dic): A dictionary that keep track of training metrics.
        values (Tuple): Elements to be added to the dictionary.
    """
    hist["train_acc"].append(values[0])
    hist["train_loss"].append(values[1])
    hist["valid_acc"].append(values[2])
    hist["valid_loss"].append(values[3])
    hist["test_acc"].append(values[4])
    hist["test_loss"].append(values[5])
    hist["bound_lin"].append(values[6][0])
    hist["bound_hyp"].append(values[6][1])
    hist["bound_kl"].append(values[6][2])
    hist["bound_mrch"].append(values[6][3])


def update_wandb(wandb, hist):
    """
    Upload values to WandB.
    Args:
        wandb (package): the weights and biases package;
        hist (dic): A dictionary that keep track of training metrics.
    """
    wandb.log({"train_acc": hist["train_acc"][-1],
               "train_loss": hist["train_loss"][-1],
               "valid_acc": hist["valid_acc"][-1],
               "valid_loss": hist["valid_loss"][-1],
               "test_acc": hist["test_acc"][-1],
               "test_loss": hist["test_loss"][-1],
               "bound_lin": hist["bound_lin"][-1],
               "bound_hyp": hist["bound_hyp"][-1],
               "bound_kl": hist["bound_kl"][-1],
               "bound_mrch": hist["bound_mrch"][-1]})
