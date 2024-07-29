import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.model.utils.loss import linear_loss


def highlight_cell(x, y, ax=None, label='none', **kwargs):
    if label == 'none':
        rect = plt.Rectangle((x - .5, y - .5), 0.95, 0.95, fill=False, **kwargs)
    else:
        rect = plt.Rectangle((x - .5, y - .5), 0.95, 0.95, fill=False, label=label, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def show_performance_matrix(meta_pred: SimpleMetaNet, pred, dataset_name, dataset, classes_name, idx, n_datasets,
                            is_using_wandb, wandb, batch_size, device):
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
        device (str): "gpu", or "cpu"; whether to use the gpu.
    """
    n_classes = int((1 + (1 + n_datasets * 4) ** 0.5) / 2)
    meta_pred.eval()
    with torch.no_grad():
        examples = []
        inputs = torch.from_numpy(dataset).float()
        integers = list(range(len(inputs[0])))
        for i in range(len(inputs)):
            random.shuffle(integers)
            inputs[i] = inputs[i, integers]
        m = int(len(inputs[0]) / 2)
        accs = np.ones((n_classes, n_classes))
        train_valid_test = np.ones((n_classes, n_classes), dtype=str)
        zs = torch.zeros((n_datasets, m))
        if str(device) == "gpu":
            inputs, meta_pred, zs = inputs.cuda(), meta_pred.cuda(), zs.cuda()
        for i in range(math.ceil(len(inputs) / batch_size)):
            first = i * batch_size
            last = min((i + 1) * batch_size, len(inputs))
            meta_output = meta_pred.forward(inputs[first:last, :m])
            pred.set_params(meta_output)
            _, z = pred.forward(inputs[first:last, :m])
            zs[first:last] = z
        k = 0
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    loss = linear_loss(zs[k], inputs[k, :m, -1])
                    if k in idx[0]:
                        train_valid_test[i, j] = "t"
                    elif k in idx[1]:
                        train_valid_test[i, j] = "v"
                    elif k in idx[2]:
                        train_valid_test[i, j] = "e"
                    accs[i, j] = loss
                    k += 1
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        if dataset_name == "mnist":
            fig, ax = plt.subplots()
        elif dataset_name == "cifar100_binary":
            fig, ax = plt.subplots(figsize=(40, 40))
        im = ax.imshow(np.transpose(accs), cmap="Greys")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(classes_name)), labels=classes_name)
        ax.set_yticks(np.arange(len(classes_name)), labels=classes_name)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        Train, Valid, Test = True, True, True
        min_accs = np.min(accs)
        for i in range(len(classes_name)):
            for j in range(len(classes_name)):
                c = 'black' if accs[i, j] < (1 + min_accs) / 2 and not i == j else 'white'
                ax.text(i, j, round(accs[i, j], 2), ha="center", va="center", color=c, size=10)
                if train_valid_test[i, j] == "t":
                    if Train == True:
                        label = "Train"
                        Train = False
                    highlight_cell(i, j, color="green", linewidth=5, label=label)
                    ax.legend(['Train'], loc="upper right", fontsize="large")
                elif train_valid_test[i, j] == "v":
                    if Valid == True:
                        label = "Valid"
                        Valid = False
                    highlight_cell(i, j, color="blue", linewidth=5, label=label)
                    ax.legend(['Valid'], loc="upper right", fontsize="large")
                elif train_valid_test[i, j] == "e":
                    if Test == True:
                        label = "Test"
                        Test = False
                    highlight_cell(i, j, color="red", linewidth=5, label=label)
                label = "none"
        ax.legend(loc="upper right", fontsize="large")
        ax.set_title(f"Performance matrix for the {dataset_name} dataset")
        fig.tight_layout()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    figure_folder_path = Path("figures")
    if not figure_folder_path.exists():
        figure_folder_path.mkdir()

    decision_boundaries_folder_name = "performance_matrix"
    decision_boundaries_folder_path = figure_folder_path / decision_boundaries_folder_name
    if not decision_boundaries_folder_path.exists():
        decision_boundaries_folder_path.mkdir()
    plt.savefig(decision_boundaries_folder_path / "performance_matrix.png")
    im_frame = Image.open(decision_boundaries_folder_path / "performance_matrix.png")
    if is_using_wandb:
        image = wandb.Image(np.array(im_frame),
                            caption=f"{decision_boundaries_folder_name}/performance_matrix.png")  # file_type="jpg"
        examples.append(image)
        wandb.log({"Decision boundaries": examples})
