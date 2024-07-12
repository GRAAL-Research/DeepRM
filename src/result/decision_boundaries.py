from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from src.model.predictor import Predictor
from src.model.utils.loss import lin_loss


def show_decision_boundaries(meta_pred, dataset, data_loader, pred: Predictor, wandb, device):
    """
    Builds a visual depiction of the decision boundary of the predictor for tackling a given problem.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train;
        dataset (str): name of the current dataset;
        data_loader (DataLoader): A DataLoader to test on;
        wandb (package): the weights and biases package;
        device (str): "gpu", or "cpu"; whether to use the gpu.
    """
    max_number_vis = 16  # Maximum number of decision boundaries to compute
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        examples = []
        for inputs in data_loader:
            for j in range(len(inputs)):
                if  i < max_number_vis:
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()
                    i += 1
                    targets = (inputs.clone()[:, :, -1] + 1) / 2
                    inputs, targets = inputs.float(), targets.float()
                    # ... so that each class can be plotted with different colours

                    x = inputs[j:j+1]
                    m = int(len(x[0]) / 2)
                    if str(device) == "gpu":
                        inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                    meta_output = meta_pred(inputs[:, :m])[j:j + 1]
                    if pred.pred_type == "linear_classif":
                        px = [-20, 20]
                        py = [-(-20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1],
                              -(20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1]]
                        plt.plot(px, py)  # With a linea classifier, only a line needs to be drawn
                    if pred.pred_type == "small_nn":
                        # With small nn: we plot the decision boundary by colouring each decision zone by its prediction
                        h = .05  # step size in the mesh
                        x_min, x_max = x[0, :, 0].cpu().min() - 10, x[0, :, 0].cpu().max() + 10
                        y_min, y_max = x[0, :, 1].cpu().min() - 10, x[0, :, 1].cpu().max() + 10
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                             np.arange(y_min, y_max, h))
                        mesh = np.array(np.c_[xx.ravel(), yy.ravel()])
                        mesh = np.hstack((mesh, np.ones((len(mesh), 1))))
                        mesh = torch.from_numpy(np.array([mesh])).float()
                        if str(device) == "gpu":
                            mesh = mesh.cuda()
                        pred.set_weights(meta_output, 1)

                        _, z = pred.forward(x)
                        acc = lin_loss(z[0], x[0, :, -1])
                        plt.scatter(x[0, x[0, :, 2] == 1, 0].cpu(), x[0, x[0, :, 2] == 1, 1].cpu(), c="r")
                        plt.scatter(x[0, x[0, :, 2] == -1, 0].cpu(), x[0, x[0, :, 2] == -1, 1].cpu(), c="b")
                        plt.text(torch.mean(x[0, :, 0].cpu()) - 9.5,
                                 torch.mean(x[0, :, 1].cpu()) - 9.2,
                                 f"Accuracy: {round(acc.item() * 100, 2)}%",
                                 bbox=dict(fill=True, color='white', edgecolor='black', linewidth=2))

                        _, z = pred.forward(mesh, use_last_values=True)
                        z = z.reshape(xx.shape).cpu()
                        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.6)

                    if meta_pred.comp_set_size > 0:
                        meta_pred.compute_compression_set(inputs[:, :m])
                        plt.scatter(x[meta_pred.msk[j].cpu(), 0].cpu(),
                                    x[meta_pred.msk[j].cpu(), 1].cpu(), c="black", s=120, marker="*")
                    if dataset == "blob":
                        plt.xlim(-20, 20)
                        plt.ylim(-20, 20)
                    if dataset == "moon":
                        plt.xlim(torch.mean(x[0, :, 0].cpu()) - 10, torch.mean(x[0, :, 0].cpu()) + 10)
                        plt.ylim(torch.mean(x[0, :, 1].cpu()) - 10, torch.mean(x[0, :, 1].cpu()) + 10)

                    figure_folder_path = Path("figures")
                    if not figure_folder_path.exists():
                        figure_folder_path.mkdir()

                    decision_boundaries_folder_name = "decision_boundaries"
                    decision_boundaries_folder_path = figure_folder_path / decision_boundaries_folder_name
                    if not decision_boundaries_folder_path.exists():
                        decision_boundaries_folder_path.mkdir()

                    fig_prefix = "decision_boundaries"
                    plt.savefig(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                    if wandb is not None:
                        im_frame = Image.open(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                        image = wandb.Image(np.array(im_frame),
                                            caption=f"{decision_boundaries_folder_name}/{fig_prefix}_{i}")  # file_type="jpg"
                        examples.append(image)

    wandb.log({"Decision boundaries": examples})
