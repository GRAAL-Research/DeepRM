import random
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from src.model.predictor import Predictor
from src.model.utils.loss import lin_loss


def show_performance_matrix(meta_pred, pred, dataset, is_using_wandb, wandb, device):
    """
    Builds a visual depiction of the decision boundary of the predictor for tackling a given problem.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train;
        pred (Predictor):
        dataset (torch.Tensor): the dataset;
        is_using_wandb (bool): whether to use wandb;
        wandb (package): the weights and biases package;
        device (str): "gpu", or "cpu"; whether to use the gpu.
    """
    meta_pred.eval()
    with torch.no_grad():
        examples = []
        inputs = torch.from_numpy(dataset).float()
        integers = list(range(len(inputs[0])))
        for i in range(len(inputs)):
            random.shuffle(integers)
            inputs[i] = inputs[i, integers]
        if str(device) == "gpu":
            inputs, meta_pred = inputs.cuda(), meta_pred.cuda()
        m = int(len(inputs[0]) / 2)
        meta_output = meta_pred(inputs[:, :m])
        pred.set_weights(meta_output)
        _, z = pred.forward(inputs[:, :m])
        accs = np.ones((10, 10))
        k = 0
        for i in range(10):
            for j in range(10):
                if i != j:
                    loss = lin_loss(z[k], inputs[k, :m, -1])
                    accs[i, j] = loss
                    k += 1
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        axis_vals = list(range(10))
        fig, ax = plt.subplots()
        im = ax.imshow(accs, cmap="Greys")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(axis_vals)), labels=axis_vals)
        ax.set_yticks(np.arange(len(axis_vals)), labels=axis_vals)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(axis_vals)):
            for j in range(len(axis_vals)):
                c = 'black' if accs[i, j] < 0.7 and not i == j else 'white'
                text = ax.text(j, i, round(accs[i, j], 2), ha="center", va="center", color=c, size=10)

        ax.set_title("Performance matrix for the MNIST binary dataset")
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
