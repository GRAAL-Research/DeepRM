# Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks

### Dependencies

- Use Python 3.11
- Install the other requirements from the `requirements.txt`

### Train on toy (moon) dataset

Run `main.py`.

### Reproduce main results

To reproduce the paper's main results on real benchmarks, set the `dataset_config_path` hyperparameter to `dataset/dataset_name.yaml` in `config/config.yaml`, then run `main.py`.

### Bibtex

If you find this work useful, please cite:

```
@article{leblanc2025generalization,
  title={Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks},
  author={Leblanc, Benjamin and Bazinet, Mathieu and D'Amours, Nathaniel and Drouin, Alexandre and Germain, Pascal},
  journal={ICML},
  year={2025}
}
```
