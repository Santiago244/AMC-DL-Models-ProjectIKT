# RadioML 2018.01A PyTorch Experiments

This project trains and evaluates PyTorch implementations of three GitHub-reference automatic modulation classification models on RadioML 2018.01A:

- `CLDNNLikeModel`
- `MCLDNN`
- `MCNet`

The project follows a controlled comparison protocol: all baseline experiments reuse the same saved stratified train/validation/test split, the same random seed, the same optimizer, the same learning rate, the same batch size, the same number of epochs, and the same evaluation procedure. Under this setup, the main experimental variable is the model architecture.

The original papers for these models sometimes use different training settings. Those paper-specific settings are not used here, because the goal is a fair architecture comparison under identical training conditions.

## Dataset

The raw RadioML file should be placed at:

```text
data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5
```

The saved split file is:

```text
data/splits/radioml2018a_seed42_60_20_20.npz
```

The split contains:

```text
train_idx
val_idx
test_idx
```

The split is stratified by modulation class and SNR level:

```text
24 modulation classes x 26 SNR levels = 624 stratification groups
```

The split ratios are:

```text
Training:   60%
Validation: 20%
Testing:    20%
```

The full baseline split contains:

```text
Training:   1,533,792 samples
Validation:   511,056 samples
Testing:      511,056 samples
```

## Models

The translated PyTorch model files are stored in:

```text
github-models/
```

The references are:

```text
CLDNN-like:
https://github.com/leena201818/radioml/blob/master/rmlmodels/CLDNNLikeModel.py

MCLDNN:
https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML2018/MCLDNN/rmlmodels/MCLDNN.py

MCNet:
https://github.com/ThienHuynhThe/MCNet/blob/master/mcnet_commlett.m
```

The PyTorch versions keep the reference network designs and internal model hyperparameters. The final softmax/classification layer is represented as raw logits because `torch.nn.CrossEntropyLoss` applies the softmax operation internally.

## Configuration Files

Run the new models using:

```text
configs/cldnn_like_baseline.yaml
configs/mcldnn_baseline.yaml
configs/mcnet_baseline.yaml
```

The shared training settings are:

```text
batch_size: 256
epochs: 50
learning_rate: 0.001
optimizer: adam
weight_decay: 0.0
early_stopping_patience: 10
random_seed: 42
sample_normalize: false
```

The shared dataset setup is:

```text
split_path: data/splits/radioml2018a_seed42_60_20_20.npz
split ratio: 60/20/20
stratification: modulation class + SNR level
```

## Training

Run CLDNN-like:

```bash
python -m src.train --config configs/cldnn_like_baseline.yaml --device cuda
```

Run MCLDNN:

```bash
python -m src.train --config configs/mcldnn_baseline.yaml --device cuda
```

Run MCNet:

```bash
python -m src.train --config configs/mcnet_baseline.yaml --device cuda
```

Use `--device auto` or omit `--device` if you want the script to choose CUDA when available and CPU otherwise:

```bash
python -m src.train --config configs/mcldnn_baseline.yaml
```

Each training run creates a timestamped folder in:

```text
experiments/
```

Expected training outputs:

```text
config.yaml
effective_config.json
history.csv
train_report.json
best_model.pth
```

## Evaluation

After training, evaluate the best checkpoint on the held-out test split.

Example for CLDNN-like:

```bash
python -m src.evaluate --config configs/cldnn_like_baseline.yaml --checkpoint experiments/<cldnn_like_run>/best_model.pth
```

Example for MCLDNN:

```bash
python -m src.evaluate --config configs/mcldnn_baseline.yaml --checkpoint experiments/<mcldnn_run>/best_model.pth
```

Example for MCNet:

```bash
python -m src.evaluate --config configs/mcnet_baseline.yaml --checkpoint experiments/<mcnet_run>/best_model.pth
```

Evaluation writes:

```text
test_report.json
confusion_matrix.csv
```

The test report includes:

```text
overall_accuracy
macro_precision
macro_recall
macro_f1
per_snr_accuracy
per_class_accuracy
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
```

## Comparing Runs

After evaluating the models, generate comparison tables:

```bash
python -m src.compare_models --experiments-dir experiments --output-dir experiments
```

This writes:

```text
experiments/model_kpis_all_runs.csv
experiments/model_kpis_all_runs.md
experiments/model_kpis_best_per_model.csv
experiments/model_kpis_best_per_model.md
```

## MATLAB Visualizations

The MATLAB plotting script reads one experiment directory at a time:

```text
visualizations/plot_current_experiment.m
```

Change `experimentDir` inside the MATLAB script to the experiment folder you want to visualize, for example:

```matlab
experimentDir = fullfile(projectRoot, "experiments", "mcldnn_baseline_YYYYMMDD_HHMMSS");
```

Then run:

```matlab
cd 'C:\Users\GSSANTIAGO\SANTIAGO GS\UNIZA\FORTH-SEMESTER\projekt-z-IKT'
run('visualizations/plot_current_experiment.m')
```

The script exports PNG figures such as learning curves, summary metrics, accuracy versus SNR, per-class metrics, confusion matrices, prediction bias, and SNR-regime accuracy.

## Notes

- The dataset split, random seed, and split percentages are unchanged.
- The training loop, evaluation script, and MATLAB visualization pipeline are unchanged.
- The original CNN1D, CNN-LSTM, and ResNet1D code is still present in the repository, but the run instructions above focus on the new GitHub-reference models.
