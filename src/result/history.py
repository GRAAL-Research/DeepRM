from src.utils.utils import TEST_ACCURACY, TRAIN_ACCURACY, VALID_ACCURACY, TEST_LOSS, \
    TRAIN_LOSS, VALID_LOSS, LINEAR_BOUND, HYP_BOUND, KL_BOUND, MARCHAND_BOUND


def update_hist(hist, values):
    """
    Adds values to the hist. dictionary to keep track of the losses and accuracies for each epoch.
    Args:
        hist (dic): A dictionary that keep track of training metrics.
        values (Tuple): Elements to be added to the dictionary.
    """
    hist[TRAIN_ACCURACY].append(values[0])
    hist[TRAIN_LOSS].append(values[1])
    hist[VALID_ACCURACY].append(values[2])
    hist[VALID_LOSS].append(values[3])
    hist[TEST_ACCURACY].append(values[4])
    hist[TEST_LOSS].append(values[5])
    hist[LINEAR_BOUND].append(values[6][0])
    hist[HYP_BOUND].append(values[6][1])
    hist[KL_BOUND].append(values[6][2])
    hist[MARCHAND_BOUND].append(values[6][3])


def update_wandb(wandb, hist):
    """
    Upload values to WandB.
    Args:
        wandb (package): the weights and biases package;
        hist (dic): A dictionary that keep track of training metrics.
    """
    wandb.log({TRAIN_ACCURACY: hist[TRAIN_ACCURACY][-1],
               TRAIN_LOSS: hist[TRAIN_LOSS][-1],
               VALID_ACCURACY: hist[VALID_ACCURACY][-1],
               VALID_LOSS: hist[VALID_LOSS][-1],
               TEST_ACCURACY: hist[TEST_ACCURACY][-1],
               TEST_LOSS: hist[TEST_LOSS][-1],
               LINEAR_BOUND: hist[LINEAR_BOUND][-1],
               HYP_BOUND: hist[HYP_BOUND][-1],
               KL_BOUND: hist[KL_BOUND][-1],
               MARCHAND_BOUND: hist[MARCHAND_BOUND][-1]})
