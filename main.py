from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

from datasets import *
from train import *


def deeprm(experiment_name, param_grid, dataset):
    """
    Args:
        experiment_name: (list of str) experiment name;
        dataset (list of str): datasets (choices: 'moons', 'easy', 'both');
        seed (list of int): random seeds;
        n (list of int): total number of datasets
        m (list of int): number of examples per dataset;
        d (list of int): dimension of each example;
        splits (list of [float, float, float]): train, valid and test proportion of the data;
        meta_pred (list of str): meta_predictor to use (choices: 'simplenet');
        pred_arch (list of list of int): architecture of the predictor (a ReLU network);
        comp_set_size (list of int): compression set size;
        msg_size (list of int): message size;
        ca_dim (list of list of int): custom attention's MLP architecture;
        kme_dim (list of list of int): KME's MLP architecture;
        mod_1_dim (list of list of int): MLP #1 architecture;
        mod_2_dim (list of list of int): MLP #2 architecture;
        tau (list of int): temperature parameter (softmax in custom attention);
        init (list of str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
        criterion (list of str): loss function (choices: 'bce_loss');
        start_lr (list of float): initial learning rate;
        pen_msg (list of str): type of message penalty (choices: 'l1', 'l2');
        pen_msg_coef (list of float): message penalty coefficient;
        msg_type (list of str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
        batch_size (list of int): batch size;
        scheduler (list of str): learning rate decay type (choices: 'plateau');
        patience (list of int): learning rate decay patience;
        factor (list of float): factor by which the learning rate is multiplied (one decay step);
        tol (list of float): tolerance for learning rate decay to apply;
        early_stop (list of int): early stopping number of epoch;
        optimizer (list of str): meta-neural network optimizer (choices: 'adam', 'rmsprop');
        n_epoch (list of int): maximum number of epoch for the training phase;
        device (list of str): device on which to compute (choices: 'cpu', 'gpu');
        weightsbiases (list of [str, str]): list with WandB team and project if data is to be stocked on WandB;
                      (empty list): if data is not to be stocked in WandB.
    """
    # Creating a parameter grid for iterating on the different hyperparameters combinations.
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset)}
    param_grid = sorted(param_grid, key=lambda p: ordering[p['dataset']])
    n_tasks = len(param_grid)
    meta_pred, data, opti, sched, crit, penalty_msg, task_dict = None, None, None, None, None, None, None

    for i, task_dict in enumerate(param_grid):  # Iterating on the different hyperparameters combinations.
        if task_dict['dataset'] == 'mnist':
            task_dict['n'], task_dict['m'], task_dict[
                'd'] = 90, 6313 * 2, 784  # For non-synthetic data, these are fixed
        if task_dict['msg_size'] == 0:
            task_dict['msg_type'] = 'none'
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        if task_dict['msg_type'] == 'dsc' and task_dict['pen_msg_coef'] > 0:  # Passes on incoherent hyp. param. comb.
            print("Doesn't make sens to regularize discrete messages; passing...\n")
        elif task_dict['comp_set_size'] + task_dict['msg_size'] == 0:  # Passes on incoherent hyp. param. comb.
            print("Opaque network; passing...\n")
        elif is_job_already_done(experiment_name, task_dict):  # Verify if the hyp. param. comb. has already been tested
            print("Already done; passing...\n")
        else:  # The current hyp. param. comb. will be tested
            set_seed(task_dict['seed'])  # Sets the random seed for numpy, torch and random packages
            if task_dict['dataset'] in ['moons', 'easy', 'hard']:  # Generating the datasets
                data = data_gen(task_dict['dataset'], task_dict['n'], task_dict['m'], task_dict['d'], True)
            elif task_dict['dataset'] == 'mnist':
                data = load_mnist()
            pred = Predictor(task_dict['d'], task_dict['pred_arch'], task_dict['batch_size'])  # Predictor init.
            if task_dict['meta_pred'] == 'simplenet':  # Meta-predictor initialization
                meta_pred = SimpleMetaNet(pred.num_param, task_dict)
            if task_dict['criterion'] == 'bce_loss':  # Criterion initialization
                crit = nn.BCELoss(reduction='none')
            if task_dict['pen_msg'] == 'l1':  # Message penalty initialization
                penalty_msg = l1
            elif task_dict['pen_msg'] == 'l2':
                penalty_msg = l2
            if task_dict['optimizer'] == 'adam':  # Optimizer initialization
                opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            elif task_dict['optimizer'] == 'rmsprop':
                opti = torch.optim.RMSprop(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            if task_dict['scheduler'] == 'plateau':  # Scheduler initialization
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti, mode='max',
                                                                   factor=task_dict['factor'],
                                                                   patience=task_dict['patience'],
                                                                   threshold=task_dict['tol'], verbose=True)
            hist, best_epoch = train(meta_pred, pred, data, opti, sched, crit, penalty_msg, task_dict)  # Launch train.
            write(experiment_name, task_dict, hist, best_epoch)  # Training details are written in a .txt file


if __name__ == "__main__":
    experiment_name = "DeepRM - message exp"
    omega_config = OmegaConf.load("config.yaml")
    config = OmegaConf.to_container(omega_config, resolve=True)
    param_grid = ParameterGrid([config])
    dataset = config["dataset"]
    deeprm(experiment_name, param_grid, dataset)
