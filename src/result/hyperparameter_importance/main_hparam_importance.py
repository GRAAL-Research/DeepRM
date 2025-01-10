from src.result.hyperparameter_importance.download_and_save_wandb_data import fetch_wandb_data
from src.result.hyperparameter_importance.show_correlation_with_test_accuracy import show_correlation_with_test_accuracy
from src.result.hyperparameter_importance.show_gini_importance import show_gini_importance
from src.result.hyperparameter_importance.show_permutation_importance import show_permutation_importance
from src.result.hyperparameter_importance.show_shap_values import show_shap_values
from src.result.hyperparameter_importance.show_two_hparams_impact_on_test_accuracy import \
    show_two_hparams_impact_on_test_accuracy

if __name__ == '__main__':
    team = "graal-deeprm2024"
    project = "message-module-exp6-mnist"  # or "message-module-with-kme-exp5-mnist"
    figure_file_name = f"{team}_{project}"

    data = fetch_wandb_data(team, project)
    show_correlation_with_test_accuracy(data, figure_file_name)

    hyperparameters = ["lr", "meta_batch_size", "msg_size", "tfm_input_dim", "tfm_n_heads", "tfm_mlp_dim", "tfm_n_encoders",
                       "tfm_drop_out", "tfm_output_dim"]
    show_two_hparams_impact_on_test_accuracy(data, hyperparameters)
    show_gini_importance(data)
    show_permutation_importance(data)
    show_shap_values(data, team, project, n_instances=5_000)
