# Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks

This repository contains the source code of experiments presented in  "[Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks](https://arxiv.org/abs/2410.13577)" by Leblanc, Bazinet, D'Amours, Drouin and Germain, accepted at ICML 2025.

### Dependencies

- Use Python 3.11
- Install the other requirements from the `requirements.txt`

### Train on toy (moon) dataset

Run `main.py`.

### Reproducing the main results

To reproduce the paper's main results on real benchmarks, set the `dataset_config_path` hyperparameter to `dataset/dataset_name.yaml` in `config/config.yaml`, then run `main.py`.

### Saving the runs in Weights & Biases

Set the `log_config_path` hyperparameter to `log/on.yaml` in `config/config.yaml`.

### Managing the other hyperparameters

The files containing the hyperparameters impacting the runs are mostly found in `config/config.yaml` and `config/grid_search_override.yaml`. Depending on the chosen task, whether the run is saved in Weights & Biases, and the chosen device, other hyperparameters are found in the corresponding .yaml files in `config/dataset`, `config/log`, and `config/training`. Every hyperparameter can also be present in `config/grid_search_override.yaml` with a list of values to test; these values override the ones present in other yaml files, if so (in other words: `config/grid_search_override.yaml` has priority the other .yaml files).

Take a look at the `hyperparameters_dictionary.txt` to learn more about the effect of many tunable hyperparameters.

### BiBTeX

If you find this work useful, please cite:

```
@article{leblanc2025generalization,
  title={Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks},
  author={Leblanc, Benjamin and Bazinet, Mathieu and D'Amours, Nathaniel and Drouin, Alexandre and Germain, Pascal},
  journal={ICML},
  year={2025}
}
```
