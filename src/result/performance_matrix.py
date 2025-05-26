import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.model.utils.loss import linear_loss
from src.utils.utils import FIGURE_BASE_PATH


def highlight_cell(x, y, ax=None, label='none', **kwargs):
    if label == 'none':
        rect = plt.Rectangle((x - .5, y - .5), 0.95, 0.95, fill=False, **kwargs)
    else:
        rect = plt.Rectangle((x - .5, y - .5), 0.95, 0.95, fill=False, label=label, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def compute_acc(n_classes, idx, m, inputs, outputs, accs, are_test_classes_shared_with_train):
    t_v_e_matrix = np.ones((n_classes, n_classes), dtype=str)
    k = 0
    train_acc, train_cnt = 0, 0
    valid_acc, valid_cnt = 0, 0
    test_with_acc, test_with_cnt = 0, 0
    test_without_acc, test_without_cnt = 0, 0
    n_test = -1
    n_test_without = 0
    while n_test_without < len(idx[2]):
        n_test += 1
        n_test_without = 0
        for j in range(n_test + 1):
            n_test_without += (n_classes - j - 1) * 2
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                loss = linear_loss(torch.reshape(outputs[[k]], (1, -1)), inputs[[k], m:, -1])
                accs[i, j] = loss
                if k in idx[0]:
                    t_v_e_matrix[i, j] = "t"
                    train_acc += accs[i, j]
                    train_cnt += 1
                elif k in idx[1]:
                    t_v_e_matrix[i, j] = "v"
                    valid_acc += accs[i, j]
                    valid_cnt += 1
                elif k in idx[2]:
                    t_v_e_matrix[i, j] = "e"
                    if i > n_test or j > n_test or are_test_classes_shared_with_train:
                        test_with_acc += accs[i, j]
                        test_with_cnt += 1
                    else:
                        test_without_acc += accs[i, j]
                        test_without_cnt += 1
                k += 1
    train_acc /= train_cnt
    valid_acc /= valid_cnt
    test_with_acc /= test_with_cnt
    if not are_test_classes_shared_with_train:
        test_without_acc /= test_without_cnt
    return train_acc, valid_acc, test_with_acc, test_without_acc, t_v_e_matrix


def show_performance_matrix(meta_pred: SimpleMetaNet, pred, dataset_name, dataset, classes_name, idx, n_datasets,
                            is_using_wandb, wandb, meta_batch_size, are_test_classes_shared_with_train, device):
    """
    Builds a visual depiction of the decision boundary of the predictor for tackling a given problem.
    Args:
        batch_size:
        meta_pred (nn.Module): A meta predictor (neural network) to train;
        pred (Predictor):
        dataset_name:
        dataset (ndarray): the dataset;
        classes_name:
        idx:
        n_datasets:
        is_using_wandb (bool): whether to use wandb;
        wandb (package): the weights and biases package;
        are_test_classes_shared_with_train:
        device (str): "gpu", or "cpu"; whether to use the gpu.
    """

    n_classes = int((1 + (1 + n_datasets * 4) ** 0.5) / 2)
    meta_pred.eval()
    with torch.no_grad():
        examples = []
        inputs = torch.from_numpy(dataset).float()
        m = int(len(inputs[0]) / 2)
        accs = np.ones((n_classes, n_classes))
        outputs = torch.zeros((n_datasets, m))
        if str(device) == "gpu":
            inputs, meta_pred, outputs = inputs.cuda(), meta_pred.cuda(), outputs.cuda()
        for i in range(math.ceil(len(inputs) / meta_batch_size)):
            first = i * meta_batch_size
            last = min((i + 1) * meta_batch_size, len(inputs))
            meta_output = meta_pred.forward(inputs[first:last, :m])
            pred.set_params(meta_output)
            _, output = pred.forward(inputs[first:last, m:])
            outputs[first:last] = output[:, :, 0]
        train_acc, valid_acc, test_with_acc, test_without_acc, t_v_e_matrix = \
            compute_acc(n_classes, idx, m, inputs, outputs, accs, are_test_classes_shared_with_train)

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        if dataset_name == "mnist":
            fig, ax = plt.subplots()
        elif dataset_name == "cifar100":
            fig, ax = plt.subplots(figsize=(40, 40))
        im = ax.imshow(np.transpose(accs), cmap="Greys")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(classes_name)), labels=classes_name, rotation=30)
        ax.set_yticks(np.arange(len(classes_name)), labels=classes_name)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        tr_in_legend, vd_in_legend, te_in_legend = False, False, False
        linewidth = 5 * (dataset_name == "cifar100") + 2 * (dataset_name == "mnist")
        min_accs = np.min(accs)
        for i in range(len(classes_name)):
            for j in range(len(classes_name)):
                c = 'black' if accs[i, j] < (1 + min_accs) / 2 and not i == j else 'white'
                ax.text(i, j, round(accs[i, j], 2), ha="center", va="center", color=c, size=8)
                if t_v_e_matrix[i, j] == "t":
                    if not tr_in_legend:
                        label = "Train"
                        tr_in_legend = True
                    highlight_cell(i, j, color="green", linewidth=linewidth, label=label)
                    ax.legend(['Train'], loc="upper right", fontsize="large")
                elif t_v_e_matrix[i, j] == "v":
                    if not vd_in_legend:
                        label = "Valid"
                        vd_in_legend = True
                    highlight_cell(i, j, color="blue", linewidth=linewidth, label=label)
                    ax.legend(['Valid'], loc="upper right", fontsize="large")
                elif t_v_e_matrix[i, j] == "e":
                    if not te_in_legend:
                        label = "Test"
                        te_in_legend = True
                    highlight_cell(i, j, color="red", linewidth=linewidth, label=label)
                label = "none"
        ax.legend(loc="upper right", fontsize="large")
        plt.tight_layout()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
        if are_test_classes_shared_with_train:
            ax.set_title(
                f"Performance matrix for the {dataset_name} dataset\nTrain acc.: {round(train_acc, 3)} \nValid acc.: "
                f"{round(valid_acc, 3)} \nTest acc.: {round(test_with_acc, 3)}", fontsize=10)
        else:
            ax.set_title(
                f"Performance matrix for the {dataset_name} dataset\nTrain acc.: {round(train_acc, 3)} \nValid acc.: "
                f"{round(valid_acc, 3)} \nTest acc. (shared): {round(test_with_acc, 3)} \nTest acc. (not shared): "
                f"{round(test_without_acc, 3)}", fontsize=10)

    plt.tight_layout()
    if not FIGURE_BASE_PATH.exists():
        FIGURE_BASE_PATH.mkdir()

    decision_boundaries_folder_name = "performance_matrix"
    decision_boundaries_folder_path = FIGURE_BASE_PATH / decision_boundaries_folder_name
    if not decision_boundaries_folder_path.exists():
        decision_boundaries_folder_path.mkdir()
    plt.savefig(decision_boundaries_folder_path / "performance_matrix.png")
    im_frame = Image.open(decision_boundaries_folder_path / "performance_matrix.png")
    if is_using_wandb:
        image = wandb.Image(np.array(im_frame),
                            caption=f"{decision_boundaries_folder_name}/performance_matrix.png")  # file_type="jpg"
        examples.append(image)
        wandb.log({"Decision boundaries": examples})
