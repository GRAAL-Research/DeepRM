from config_utils import load_yaml_config_file, create_config_combinations_sorted_by_dataset
from datasets import *
from model.simple_meta_net import SimpleMetaNet
from train import *


def main(config_combinations: list[dict]):
    """
    Args:
        dataset (list of str): datasets (choices: 'mnist', 'moons', 'easy', 'both',
                                                  'MTPL2_frequency', 'MTPL2_severity', 'MTPL2_pure');
        m (list of int): number of examples per dataset;
        d (list of int): dimension of each example;
        splits (list of [float, float, float]): train, valid and test proportion of the data;
        meta_pred (list of str): meta_predictor to use (choices: 'simplenet');
        pred_arch (list of list of int): architecture of the predictor (a ReLU network);
        comp_set_size (list of int): compression set size;
        ca_dim (list of list of int): custom attention's MLP architecture;
        mod_1_dim (list of list of int): MLP #1 architecture;
        mod_2_dim (list of list of int): MLP #2 architecture;
        tau (list of int): temperature parameter (softmax in custom attention);
        init (list of str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
        criterion (list of str): loss function (choices: 'bce_loss');
        pen_msg (list of str): type of message penalty (choices: 'l1', 'l2');
        pen_msg_coef (list of float): message penalty coefficient;
        msg_type (list of str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
        batch_size (list of int): batch size;
        factor (list of float): factor by which the learning rate is multiplied (one decay step);
        optimizer (list of str): meta-neural network optimizer (choices: 'adam', 'rmsprop');
        n_epoch (list of int): maximum number of epoch for the training phase;z
    """
    n_tasks = len(config_combinations)
    meta_pred, data, opti, scheduler, crit, penalty_msg = None, None, None, None, None, None
    is_sending_wandb_last_run_alert = False

    for i, task_dict in enumerate(config_combinations):
        if task_dict['dataset'] == 'mnist':
            # For non-synthetic data, these are fixed
            task_dict['n_dataset'] = 90
            task_dict['m'] = 6313 * 2
            task_dict['d'] = 784
        elif task_dict['dataset'] in ['MTPL2_frequency', 'MTPL2_severity', 'MTPL2_pure']:
            # For non-synthetic data, these are fixed
            task_dict['d'] = 76
            task_dict['batch_size'] = 50

        if task_dict['msg_size'] == 0:
            task_dict['msg_type'] = 'none'

        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")

        if task_dict['msg_type'] == 'dsc' and task_dict['pen_msg_coef'] > 0:  # Passes on incoherent hyp. param. comb.
            print("Doesn't make sens to regularize discrete messages; passing...\n")
        elif task_dict['comp_set_size'] + task_dict['msg_size'] == 0:  # Passes on incoherent hyp. param. comb.
            print("Opaque network; passing...\n")
        elif is_job_already_done(task_dict["project_name"],
                                 task_dict):  # Verify if the hyp. param. comb. has already been tested
            print("Already done; passing...\n")
        else:  # The current hyp. param. comb. will be tested
            set_seed(task_dict['seed'])  # Sets the random seed for numpy, torch and random packages

            if task_dict['dataset'] in ['moons', 'easy', 'hard']:  # Generating the datasets
                data = data_gen(task_dict['dataset'], task_dict['n_dataset'], task_dict['m'], task_dict['d'], True)
            elif task_dict['dataset'] == 'mnist':
                data = load_mnist()
            elif task_dict['dataset'] in ['MTPL2_frequency', 'MTPL2_severity', 'MTPL2_pure']:
                data = load_MTPL(task_dict['dataset'][6:], task_dict['n_dataset'], task_dict['m'])
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
                opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(task_dict['lr']))
            elif task_dict['optimizer'] == 'rmsprop':
                opti = torch.optim.RMSprop(meta_pred.parameters(), lr=copy(task_dict['lr']))

            if task_dict['scheduler'] == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti, mode='max',
                                                                       factor=task_dict['factor'],
                                                                       patience=task_dict['scheduler_patience'],
                                                                       threshold=task_dict['scheduler_threshold'],
                                                                       verbose=True)

            if i + 1 == n_tasks:
                is_sending_wandb_last_run_alert = True

            hist, best_epoch = train(meta_pred, pred, data, opti, scheduler, crit, penalty_msg, task_dict, is_sending_wandb_last_run_alert)
            write(task_dict["project_name"], task_dict, hist, best_epoch)  # Training details are written in a .txt file
    

if __name__ == "__main__":
    config_name = "config.yaml"

    config = load_yaml_config_file(Path("config") / config_name)
    config_combinations = create_config_combinations_sorted_by_dataset(config)

    main(config_combinations)
