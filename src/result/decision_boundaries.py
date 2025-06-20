import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt

from src.model.predictor.linear_classifier import LinearClassifier
from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.utils.utils import FIGURE_BASE_PATH

def show_decision_boundaries(config, meta_pred: SimpleMetaNet, dataset, data_loader, pred: Predictor, wandb, device):
    """
    Builds a visual depiction of the decision boundary of the predictor for tackling a given problem.
        Code to generate the figures 5 and 6 in the article
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train;
        dataset (str): name of the current dataset;
        data_loader (DataLoader): A DataLoader to test on;
        wandb (package): the weights and biases package;
        device (str): "gpu", or "cpu"; whether to use the gpu.
    """
    batch_size = config["batch_size"] if config["batch_size"] > 0 else config["n_instances_per_dataset"] + 1
    meta_pred.eval()
    with torch.no_grad():
        i = 4
        examples = []
        for meta_instances in data_loader:
            meta_instances = meta_instances[0]
            for i_batch in range((len(meta_instances[0]) // batch_size) + 1):
                inputs = meta_instances[:, i_batch * batch_size:(i_batch + 1) * batch_size]
                inputs = inputs.float()
                if config["device"] == "gpu":
                    inputs = inputs.cuda()
                    meta_pred = meta_pred.cuda()
                m = int(len(inputs[0]) / 2)
                meta_pred.forward(inputs[:, :m], is_in_test_mode=True)
                saved_msg = meta_pred.msg.clone()
                for j in range(len(inputs)):
                    if i < 5:
                        j = 2
                        plt.figure().clear()
                        plt.close()
                        plt.cla()
                        plt.clf()
                        global_msg = saved_msg.clone()
                        i += 1
                        # This first code is to generate what happens when the second message component is fixed
                        for k in range(11):
                            global_msg[:, 0] = 0.6 * k - 3
                            meta_outpt = meta_pred.forward(inputs[:, :m], is_in_test_mode=True, feed_message=global_msg)
                            x = inputs[j:j + 1]
                            meta_output = meta_outpt[j:j + 1]
                            if isinstance(pred, LinearClassifier):
                                px = [-20, 20]
                                py = [-(-20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1],
                                      -(20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1]]
                                plt.plot(px, py)  # With a linea classifier, only a line needs to be drawn
                            else:
                                # With small nn: we plot the decision boundary by colouring each decision zone by its prediction
                                h = .02  # step size in the mesh
                                x_min, x_max = x[0, :, 0].cpu().min() - 10, x[0, :, 0].cpu().max() + 10
                                y_min, y_max = x[0, :, 1].cpu().min() - 10, x[0, :, 1].cpu().max() + 10
                                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                                     np.arange(y_min, y_max, h))
                                mesh = np.array(np.c_[xx.ravel(), yy.ravel()])
                                mesh = np.hstack((mesh, np.ones((len(mesh), 1))))
                                mesh = torch.from_numpy(np.array([mesh])).float()
                                # The mesh is now clean
                                if device == "gpu":
                                    mesh = mesh.cuda()
                                # We compute the model predictions
                                pred.set_params(meta_output)
                                pred.set_forward_mode(save_bn_params=True)
                                # A first pass with the data, so that the batch norm parameters are well-valued.
                                _, z = pred.forward(x)
                                pred.reset_forward_mode()

                                # Scatter the points to classify
                                plt.scatter(x[0, x[0, :, 2] == 1, 0].cpu(), x[0, x[0, :, 2] == 1, 1].cpu(), c="r")
                                plt.scatter(x[0, x[0, :, 2] == -1, 0].cpu(), x[0, x[0, :, 2] == -1, 1].cpu(), c="b")

                                pred.set_forward_mode(use_last_values=True)
                                _, z = pred.forward(mesh)
                                pred.reset_forward_mode()

                                # Finally, we plot the decision boundary
                                z = z.reshape(xx.shape).cpu()
                                xx_mod, yy_mod = [], []
                                for i_px in range(z.shape[0]):
                                    for j_px in range(z.shape[1] - 3):
                                        if z[i_px, j_px] != z[i_px, j_px + 2]:
                                            xx_mod.append(xx[i_px, j_px])
                                            yy_mod.append(yy[i_px, j_px])
                                plt.scatter(xx_mod, yy_mod, c=[(1, 0.5 + k / 20, k / 20)] * len(xx_mod), s=7)

                            # We now plot the compression set
                            if meta_pred.compression_set_size > 0:
                                meta_pred.compute_compression_set(inputs[:, :m])
                                plt.scatter(x[0, meta_pred.msk[j].cpu(), 0].cpu(),
                                            x[0, meta_pred.msk[j].cpu(), 1].cpu(), c="black", s=120, marker="*")
                            if dataset == "blob":
                                plt.xlim(-20, 20)
                                plt.ylim(-20, 20)
                            if dataset == "moon":
                                plt.xlim(torch.mean(x[0, :, 0].cpu()) - 10, torch.mean(x[0, :, 0].cpu()) + 10)
                                plt.ylim(torch.mean(x[0, :, 1].cpu()) - 10, torch.mean(x[0, :, 1].cpu()) + 10)
                                plt.xticks(fontsize=15)
                                plt.yticks(fontsize=15)

                        if not FIGURE_BASE_PATH.exists():
                            FIGURE_BASE_PATH.mkdir()

                        decision_boundaries_folder_name = "decision_boundaries"
                        decision_boundaries_folder_path = FIGURE_BASE_PATH / decision_boundaries_folder_name
                        if not decision_boundaries_folder_path.exists():
                            decision_boundaries_folder_path.mkdir()

                        fig_prefix = "decision_boundaries"
                        plt.savefig(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                        if wandb is not None:
                            im_frame = Image.open(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                            image = wandb.Image(np.array(im_frame),
                                                caption=f"{decision_boundaries_folder_name}/{fig_prefix}_{i}")  # file_type="jpg"
                            examples.append(image)

                        plt.figure().clear()
                        plt.close()
                        plt.cla()
                        plt.clf()
                        global_msg = saved_msg.clone()
                        i += 1
                        # This first code is to generate what happens when the first message component is fixed
                        for k in range(11):
                            global_msg[:, 1] = 0.6 * k - 3
                            meta_outpt = meta_pred.forward(inputs[:, :m], is_in_test_mode=True, feed_message=global_msg)
                            x = inputs[j:j + 1]
                            meta_output = meta_outpt[j:j + 1]
                            if isinstance(pred, LinearClassifier):
                                px = [-20, 20]
                                py = [-(-20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1],
                                      -(20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1]]
                                plt.plot(px, py)  # With a linea classifier, only a line needs to be drawn
                            else:
                                # With small nn: we plot the decision boundary by colouring each decision zone by its prediction
                                h = .05  # step size in the mesh
                                x_min, x_max = x[0, :, 0].cpu().min() - 10, x[0, :, 0].cpu().max() + 10
                                y_min, y_max = x[0, :, 1].cpu().min() - 10, x[0, :, 1].cpu().max() + 10
                                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                                     np.arange(y_min, y_max, h))
                                mesh = np.array(np.c_[xx.ravel(), yy.ravel()])
                                mesh = np.hstack((mesh, np.ones((len(mesh), 1))))
                                mesh = torch.from_numpy(np.array([mesh])).float()
                                if device == "gpu":
                                    mesh = mesh.cuda()
                                pred.set_params(meta_output)

                                pred.set_forward_mode(save_bn_params=True)
                                _, z = pred.forward(x)
                                pred.reset_forward_mode()

                                plt.scatter(x[0, x[0, :, 2] == 1, 0].cpu(), x[0, x[0, :, 2] == 1, 1].cpu(), c="r")
                                plt.scatter(x[0, x[0, :, 2] == -1, 0].cpu(), x[0, x[0, :, 2] == -1, 1].cpu(), c="b")

                                pred.set_forward_mode(use_last_values=True)
                                _, z = pred.forward(mesh)
                                pred.reset_forward_mode()

                                z = z.reshape(xx.shape).cpu()
                                xx_mod, yy_mod = [], []
                                for i_px in range(z.shape[0]):
                                    for j_px in range(z.shape[1] - 3):
                                        if z[i_px, j_px] != z[i_px, j_px + 2]:
                                            xx_mod.append(xx[i_px, j_px])
                                            yy_mod.append(yy[i_px, j_px])
                                plt.scatter(xx_mod, yy_mod, c=[(1, 0.5 + k / 20, k / 20)] * len(xx_mod), s=7)

                            if meta_pred.compression_set_size > 0:
                                meta_pred.compute_compression_set(inputs[:, :m])
                                plt.scatter(x[0, meta_pred.msk[j].cpu(), 0].cpu(),
                                            x[0, meta_pred.msk[j].cpu(), 1].cpu(), c="black", s=120, marker="*")
                            if dataset == "blob":
                                plt.xlim(-20, 20)
                                plt.ylim(-20, 20)
                            if dataset == "moon":
                                plt.xlim(torch.mean(x[0, :, 0].cpu()) - 10, torch.mean(x[0, :, 0].cpu()) + 10)
                                plt.ylim(torch.mean(x[0, :, 1].cpu()) - 10, torch.mean(x[0, :, 1].cpu()) + 10)
                                plt.xticks(fontsize=15)
                                plt.yticks(fontsize=15)

                        if not FIGURE_BASE_PATH.exists():
                            FIGURE_BASE_PATH.mkdir()

                        decision_boundaries_folder_name = "decision_boundaries"
                        decision_boundaries_folder_path = FIGURE_BASE_PATH / decision_boundaries_folder_name
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
