# AMC Radio Signal Classification — Project File Documentation

**Project Purpose:** Automatic Modulation Classification (AMC) using deep learning models to classify radio signals from the RadioML 2018.01A dataset.

**Dataset:** RadioML 2018.01A — 24 modulation classes with I/Q samples (1024 complex samples per signal, SNR range -20 dB to +30 dB)

---

## Project Structure Overview

```
projekt-z-IKT/
├── src/                           # Main source code (PyTorch models)
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation & testing pipeline
│   ├── dataset.py                # Dataset loading & batching
│   ├── utils.py                  # Shared utilities
│   └── models/
│       ├── factory.py            # Model instantiation factory
│       ├── cnn1d.py              # 1D CNN architecture
│       ├── cnn_lstm.py           # CNN-LSTM hybrid architecture
│       └── resnet1d.py           # ResNet1D architecture
├── data/
│   ├── dataset.py                # Split index creation utility
│   ├── raw/
│   │   └── radioml2018/          # Original RadioML dataset
│   └── splits/                   # Train/val/test indices (NPZ)
├── configs/                       # YAML experiment configurations
│   ├── cnn1d_quick.yaml
│   ├── cnn1d_baseline.yaml
│   ├── cnn_lstm_quick.yaml
│   ├── cnn_lstm_baseline.yaml
│   ├── cnn_large_filters.yaml
│   ├── cnn_low_dropout.yaml
│   ├── resnet1d_quick.yaml
│   └── resnet1d_baseline.yaml
├── claude-models-tensorflow/     # Legacy TensorFlow models (reference)
├── notebooks/
│   └── exploration.ipynb         # Data exploration notebook
├── experiments/                  # Training outputs
├── visualizations/               # MATLAB plots
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # Multi-container orchestration
└── README.md                     # Project overview
```

---

# CORE TRAINING & EVALUATION MODULES

## 1. [src/train.py](src/train.py) — Training Pipeline

**Purpose:** Main training loop with early stopping, checkpoint management, and metrics tracking.

### Key Methods:

| Method                                                               | Module/Class                            | Description                                                                                                            |
| -------------------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `apply_runtime_overrides(config, epochs, batch_size, patience, ...)` | N/A                                     | Applies CLI overrides to YAML config (epochs, batch size, early stopping patience, max samples)                        |
| `build_optimizer(model, config)`                                     | `torch.optim.Adam`, `torch.optim.AdamW` | Creates optimizer (Adam or AdamW) with learning rate and weight decay from config                                      |
| `run_epoch(model, loader, criterion, device, optimizer, desc)`       | `torch.nn.Module`, `tqdm`               | Executes one training/validation epoch; returns dict with loss & accuracy metrics                                      |
| `save_history_csv(history, path)`                                    | `csv.DictWriter`                        | Saves training history (epoch, train_loss, train_accuracy, val_loss, val_accuracy) to CSV                              |
| `train(config_path, device_name, epochs, batch_size, ...)`           | Main function                           | Core training function: loads config, creates model, trains with validation, applies early stopping, saves checkpoints |

### Key Imports:

- `torch` — PyTorch tensor operations & models
- `torch.nn` — Neural network layers (CrossEntropyLoss)
- `tqdm` — Progress bars
- `pathlib.Path` — File path handling
- `yaml` (via `src.utils.load_config`) — Config loading
- `src.models.factory.build_model` — Model instantiation
- `src.dataset.build_dataloaders` — DataLoader creation
- `src.utils` — Helper functions (set_seed, get_device, create_run_dir, count_parameters)

### Workflow:

1. Load YAML config, apply CLI overrides
2. Set random seed for reproducibility
3. Create run directory with timestamped experiment folder
4. Build model, criterion (CrossEntropyLoss), optimizer
5. For each epoch:
   - Run training epoch (forward pass, backward pass, optimizer step)
   - Run validation epoch (inference only)
   - Save metrics to CSV
   - Check if validation accuracy improved → save checkpoint
   - Check early stopping patience
6. Save final training report (JSON)

---

## 2. [src/evaluate.py](src/evaluate.py) — Evaluation & Testing Pipeline

**Purpose:** Load a trained checkpoint and evaluate it on the test set with detailed metrics (accuracy, precision, recall, F1, per-SNR accuracy).

### Key Methods:

| Method                                                            | Module/Class  | Description                                                                                                    |
| ----------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `compute_report(targets, predictions, snrs, class_names)`         | `numpy`       | Computes confusion matrix, overall accuracy, per-class accuracy, precision, recall, F1, per-SNR accuracy       |
| `evaluate(config_path, checkpoint_path, output_dir, device_name)` | Main function | Loads config & checkpoint, runs inference on test set, generates test report (JSON) and confusion matrix (CSV) |

### Key Imports:

- `torch` — Model loading and inference
- `numpy` — Metrics computation (confusion matrix, precision, recall, F1)
- `tqdm` — Progress bars
- `pathlib.Path` — File I/O
- `src.dataset.build_dataloaders, CLASSES` — Test loader creation
- `src.models.factory.build_model` — Model loading
- `src.utils` — Helper functions

### Output Files:

- `test_report.json` — Contains confusion matrix, overall/per-class accuracy, per-SNR accuracy, precision, recall, F1
- `confusion_matrix.csv` — Raw confusion matrix (24×24)

---

# DATASET MANAGEMENT

## 3. [src/dataset.py](src/dataset.py) — Dataset Loading & PyTorch Integration

**Purpose:** PyTorch Dataset/DataLoader interface for RadioML 2018.01A with on-the-fly HDF5 reading (avoids loading full 25 GB into memory).

### Key Constants:

| Constant             | Value                                              | Description                                    |
| -------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `CLASSES`            | 24-element list                                    | Modulation class names (OOK, 4ASK, ..., OQPSK) |
| `DEFAULT_HDF5_PATH`  | `data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5` | Location of full dataset                       |
| `DEFAULT_SPLIT_PATH` | `data/splits/radioml2018a_seed42_60_20_20.npz`     | Location of split indices                      |

### Key Methods:

| Method                                                                                                    | Module/Class                                  | Description                                                                                      |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `read_labels_and_snr(hdf5_path)`                                                                          | `h5py.File`                                   | Reads modulation labels & SNR values from HDF5 without loading signals (used for split creation) |
| `stratified_train_val_test_split(indices, stratify_key, train_split, val_split, test_split, random_seed)` | `numpy.random`                                | Creates stratified train/val/test splits (preserves class & SNR distribution)                    |
| `_validate_split_ratios(train_split, val_split, test_split)`                                              | N/A                                           | Validates that splits sum to 1.0                                                                 |
| `build_dataloaders(config)`                                                                               | `torch.utils.data.DataLoader, RadioMLDataset` | Creates train/val/test DataLoaders from config                                                   |
| `RadioMLDataset.__init__(hdf5_path, indices, max_samples, normalize)`                                     | `torch.utils.data.Dataset`                    | Custom Dataset class that reads HDF5 on-demand                                                   |
| `RadioMLDataset.__getitem__(idx)`                                                                         | `h5py.File`                                   | Returns (signal, label, snr) tuple for a given index                                             |

### Key Imports:

- `h5py` — HDF5 file reading
- `numpy` — Array operations & random splitting
- `torch.utils.data.DataLoader, Dataset` — PyTorch data loading infrastructure
- `pathlib.Path` — File handling
- `json` — Split metadata

### Dataset Details:

- **Input shape per sample:** (1024, 2) — 1024 complex I/Q samples (stored as real/imaginary)
- **Output:** (signal_array, class_label, snr_value)
- **Stratification:** By modulation class AND SNR level (ensures balanced per-SNR evaluation)
- **Memory efficiency:** HDF5 accessed on-demand; only split indices (NPZ) stored locally

---

## 4. [data/dataset.py](data/dataset.py) — Split Index Creation Utility

**Purpose:** Create and cache train/val/test split indices from RadioML dataset with stratification.

### Key Methods:

| Method                                                                                                           | Module/Class               | Description                                                         |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `read_labels_and_snr(hdf5_path)`                                                                                 | `h5py.File`                | Extracts modulation labels & SNR values from HDF5                   |
| `stratified_train_val_test_split(indices, stratify_key, ...)`                                                    | `numpy.random.default_rng` | Deterministic stratified split (preserves class/SNR distribution)   |
| `_validate_split_ratios(train_split, val_split, test_split)`                                                     | N/A                        | Validates split ratios sum to 1.0                                   |
| `create_split_indices(hdf5_path, output_path, train_split, val_split, test_split, random_seed, stratify_by_snr)` | Main function              | Creates and saves train/val/test indices (NPZ format) with metadata |

### Key Imports:

- `h5py` — Read original HDF5 dataset
- `numpy` — Stratified splitting
- `json` — Save metadata
- `pathlib.Path` — File I/O

### Output:

- NPZ file containing `train_idx`, `val_idx`, `test_idx` (NumPy arrays of indices)
- JSON metadata: split ratios, SNR levels, sample counts, random seed, stratification method

---

# MODEL ARCHITECTURES

## 5. [src/models/factory.py](src/models/factory.py) — Model Factory

**Purpose:** Central factory function to instantiate models from config dictionary (prevents hardcoded model selection).

### Key Methods:

| Method                | Module/Class | Description                                                                                          |
| --------------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| `build_model(config)` | N/A          | Routes to CNN1D, CNNLSTM, or ResNet1D based on `config['model']['type']`; unpacks kwargs from config |

### Key Imports:

- `torch.nn.Module` — Base class for all models
- `src.models.cnn1d.CNN1D` — 1D CNN model
- `src.models.cnn_lstm.CNNLSTM` — CNN-LSTM hybrid model
- `src.models.resnet1d.ResNet1D` — ResNet1D model

---

## 6. [src/models/cnn1d.py](src/models/cnn1d.py) — 1D CNN Baseline

**Purpose:** Straightforward 1D convolutional neural network for AMC (baseline model).

### Architecture Layers:

```
Features (Sequential):
  Conv1D(input_channels, filters, kernel_size=3, padding=1)
    ↓ BatchNorm1d → ReLU → Dropout
  Conv1D(filters, filters, kernel_size=3, padding=1)
    ↓ BatchNorm1d → ReLU → Dropout
  MaxPool1d(2)
  Conv1D(filters, filters*2, kernel_size=3, padding=1)
    ↓ BatchNorm1d → ReLU → Dropout
  MaxPool1d(2)

Classifier (Sequential):
  AdaptiveAvgPool1d(1) → Flatten
  Linear(filters*2, dense_units) → ReLU → Dropout
  Linear(dense_units, num_classes)
```

### Key Methods:

| Method                                                                              | Module/Class      | Description                                                    |
| ----------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| `__init__(num_classes, input_channels, filters, kernel_size, dropout, dense_units)` | N/A               | Initializes convolutional feature extractor & dense classifier |
| `forward(x)`                                                                        | `torch.nn.Module` | Passes input through features → classifier                     |

### Default Parameters:

- `num_classes: 24` — RadioML 24 modulations
- `input_channels: 2` — I/Q channels
- `filters: 64` — Initial filter count
- `kernel_size: 3` — Conv kernel size
- `dropout: 0.5` — Dropout rate
- `dense_units: 256` — Hidden layer units

### Key Imports:

- `torch.nn` — Conv1d, BatchNorm1d, MaxPool1d, Linear, ReLU, Dropout

---

## 7. [src/models/cnn_lstm.py](src/models/cnn_lstm.py) — CNN-LSTM Hybrid

**Purpose:** Hybrid architecture combining 1D convolutions for feature extraction + LSTM for temporal sequence modeling.

### Architecture Layers:

```
Conv Block (Sequential):
  Conv1d(input_channels, conv_filters, kernel_size=5, padding=2)
    ↓ BatchNorm1d → ReLU → MaxPool1d(2) → Dropout
  Conv1d(conv_filters, conv_filters*2, kernel_size=5, padding=2)
    ↓ BatchNorm1d → ReLU → MaxPool1d(2) → Dropout

LSTM:
  Input: (batch, seq_len, conv_filters*2)
  Hidden: lstm_hidden, Layers: lstm_layers, Bidirectional: optional
  Output: (batch, seq_len, lstm_hidden * directions)

Classifier (Sequential):
  LayerNorm(lstm_hidden * directions)
  Dropout
  Linear(lstm_hidden * directions, num_classes)
```

### Key Methods:

| Method                                                                                                               | Module/Class      | Description                                                   |
| -------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------- |
| `__init__(num_classes, input_channels, conv_filters, kernel_size, lstm_hidden, lstm_layers, bidirectional, dropout)` | N/A               | Initializes CNN feature extractor, LSTM encoder, & classifier |
| `forward(x)`                                                                                                         | `torch.nn.Module` | Conv → transpose → LSTM → mean pooling → classifier           |

### Default Parameters:

- `conv_filters: 64` — Initial CNN filter count
- `kernel_size: 5` — Conv kernel
- `lstm_hidden: 128` — LSTM hidden state size
- `lstm_layers: 1` — Number of LSTM layers
- `bidirectional: True` — Bidirectional LSTM
- `dropout: 0.4` — Dropout rate

### Key Imports:

- `torch.nn` — Conv1d, LSTM, LayerNorm, Linear, Dropout, BatchNorm1d, ReLU, MaxPool1d

### Temporal Modeling:

- LSTM processes CNN-extracted features sequentially
- Bidirectional option captures context from both directions
- Mean pooling over time step dimension → single feature vector for classification

---

## 8. [src/models/resnet1d.py](src/models/resnet1d.py) — ResNet1D with Skip Connections

**Purpose:** Deep residual network adapted for 1D signals (radio) with BasicBlock residual connections.

### Architecture:

```
Stem:
  Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3)
    ↓ BatchNorm1d → ReLU → MaxPool1d(3, stride=2)

Layer1: BasicBlock × layers[0] (stride=1)
  Residual blocks with skip connections

Layer2: BasicBlock × layers[1] (stride=2, doubles channels)
  Downsampling via stride

Layer3: BasicBlock × layers[2] (stride=2, doubles channels)
  Further downsampling

Head:
  AdaptiveAvgPool1d(1) → Flatten
  Dropout
  Linear(base_channels*4, num_classes)
```

### Key Classes:

| Class          | Description                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `BasicBlock1D` | Single residual block: Conv → BN → ReLU → Dropout → Conv → BN; skip connection via identity or 1×1 conv |
| `ResNet1D`     | Main model; stacks BasicBlocks into layers with optional downsampling                                   |

### Key Methods:

| Method                                                                           | Module/Class | Description                                                                                    |
| -------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| `BasicBlock1D.__init__(in_channels, out_channels, stride, dropout)`              | N/A          | Initializes two Conv1d layers + optional shortcut (1×1 conv if stride > 1 or channel mismatch) |
| `BasicBlock1D.forward(x)`                                                        | N/A          | Applies residual connection: `out = conv_block(x) + shortcut(x)`                               |
| `ResNet1D.__init__(num_classes, input_channels, base_channels, layers, dropout)` | N/A          | Initializes stem + 3 layers of BasicBlocks                                                     |
| `ResNet1D._make_layer(out_channels, blocks, stride, dropout)`                    | N/A          | Constructs sequential layer with specified number of BasicBlocks                               |
| `ResNet1D.forward(x)`                                                            | N/A          | Stem → Layer1 → Layer2 → Layer3 → Head                                                         |

### Default Parameters:

- `base_channels: 64` — Initial channel count
- `layers: (2, 2, 2)` — Number of BasicBlocks per layer (tuple)
- `dropout: 0.1` — Dropout rate (per block)

### Key Imports:

- `torch.nn` — Conv1d, BatchNorm1d, MaxPool1d, ReLU, Dropout, Linear, AdaptiveAvgPool1d, Module, Sequential

### Skip Connections:

- Identity shortcut if stride=1 and channel counts match
- 1×1 conv shortcut if stride>1 or channels increase
- Enables training of very deep networks by gradient flow

---

# UTILITIES & CONFIGURATION

## 9. [src/utils.py](src/utils.py) — Shared Helper Functions

**Purpose:** Reusable utility functions for configuration, seeding, device management, and model inspection.

### Key Methods:

| Method                                | Module/Class                                        | Description                                                                                 |
| ------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `load_config(config_path)`            | `yaml.safe_load`                                    | Loads YAML config file → dict                                                               |
| `save_json(payload, path)`            | `json.dump`                                         | Saves dict to JSON file with 2-space indent                                                 |
| `set_seed(seed)`                      | `random.seed, numpy.random.seed, torch.manual_seed` | Sets global seed (Python, NumPy, PyTorch CPU & CUDA) for reproducibility                    |
| `get_device(preferred)`               | `torch.device`                                      | Returns CUDA device if available; falls back to CPU; accepts "auto" or explicit device name |
| `create_run_dir(config, config_path)` | `shutil.copy2, datetime`                            | Creates timestamped experiment directory & copies config.yaml                               |
| `count_parameters(model)`             | `torch.nn.Module.parameters`                        | Counts total trainable parameters in model                                                  |

### Key Imports:

- `yaml` — YAML parsing
- `json` — JSON serialization
- `random`, `numpy.random`, `torch` — Seeding
- `pathlib.Path` — File paths
- `datetime` — Timestamp generation
- `shutil` — File operations

---

## 10. Configuration Files (YAML)

**Purpose:** Define model architecture, training hyperparameters, and data settings without code changes.

### Example: [configs/cnn1d_quick.yaml](configs/cnn1d_quick.yaml)

```yaml
experiment_name: "cnn1d_quick"

model:
  type: "cnn1d" # Model type (cnn1d, cnn_lstm, resnet1d)
  num_classes: 24
  input_channels: 2 # I/Q channels
  filters: 32 # Conv filter count (can be 32, 64, 128...)
  kernel_size: 3
  dropout: 0.3
  dense_units: 128

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam" # adam or adamw
  weight_decay: 0.0
  early_stopping_patience: 3 # Stop if val_acc doesn't improve for N epochs
  num_workers: 0
  pin_memory: false

data:
  hdf5_path: "data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5"
  split_path: "data/splits/radioml2018a_seed42_60_20_20.npz"
  random_seed: 42
  sample_normalize: false
  max_train_samples: 60000 # Limit for quick experiments
  max_val_samples: 12000
  max_test_samples: 12000

output:
  experiments_dir: "experiments" # Output directory for checkpoints & metrics
```

### Config Variations:

- `cnn1d_quick.yaml` — Smaller model, fewer samples (fast prototyping)
- `cnn1d_baseline.yaml` — Full dataset, balanced hyperparameters
- `cnn_large_filters.yaml` — Increased filter counts (64→128)
- `cnn_low_dropout.yaml` — Reduced dropout (0.5→0.2)
- `cnn_lstm_quick.yaml`, `cnn_lstm_baseline.yaml` — CNN-LSTM variants
- `resnet1d_quick.yaml`, `resnet1d_baseline.yaml` — ResNet1D variants

---

# EXPERIMENT OUTPUT STRUCTURE

## 11. Experiment Directories (Auto-Generated)

**Location:** `experiments/{experiment_name}_{YYYYMMDD_HHMMSS}/`

### Contents:

| File                    | Format  | Description                                                                                      |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `config.yaml`           | YAML    | Copy of original config file                                                                     |
| `effective_config.json` | JSON    | Final config after CLI overrides                                                                 |
| `best_model.pth`        | PyTorch | Checkpoint with `model_state_dict`, `optimizer_state_dict`, `epoch`, best metrics, config        |
| `history.csv`           | CSV     | Row per epoch: epoch, train_loss, train_accuracy, val_loss, val_accuracy                         |
| `train_report.json`     | JSON    | Summary: model_type, num_parameters, best_epoch, best_val_accuracy, history                      |
| `test_report.json`      | JSON    | Evaluation metrics: overall_accuracy, per_class_accuracy, per_SNR_accuracy, confusion_matrix, F1 |
| `confusion_matrix.csv`  | CSV     | 24×24 confusion matrix (rows=true, cols=predicted)                                               |

---

# LEGACY TENSORFLOW MODELS (Reference)

## 12. [claude-models-tensorflow/cnn1d.py](claude-models-tensorflow/cnn1d.py) — TensorFlow CNN1D

**Purpose:** Reference implementation of 1D CNN in TensorFlow/Keras (not used in current PyTorch pipeline).

### Key Methods:

| Method                                                | Module/Class                        | Description                                                                                  |
| ----------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `build_cnn1d(input_shape, num_classes, dropout_rate)` | `keras.Sequential` or `keras.Model` | Constructs CNN1D using Keras functional API with Conv1D, MaxPooling, BatchNorm, Dense layers |

### Key Imports:

- `tensorflow.keras` — Model building
- `keras.layers` — Conv1D, Dense, MaxPooling1D, BatchNormalization, Dropout
- `keras.regularizers` — L2 regularization

### Status:

- Reference architecture only; not integrated with current PyTorch pipeline
- Uses Keras functional API (more flexible than Sequential)

---

## 13. [claude-models-tensorflow/cnn_lstm.py](claude-models-tensorflow/cnn_lstm.py), [resnet.py](claude-models-tensorflow/resnet.py)

**Purpose:** Legacy CNN-LSTM and ResNet implementations in TensorFlow.

**Status:** Not actively used; kept for reference.

---

# NOTEBOOKS & VISUALIZATION

## 14. [notebooks/exploration.ipynb](notebooks/exploration.ipynb)

**Purpose:** Exploratory data analysis notebook (Jupyter).

**Likely Contents:**

- Load RadioML dataset
- Visualize I/Q samples, spectrograms
- Analyze class distribution, SNR levels
- Compute dataset statistics

---

## 15. [visualizations/plot_current_experiment.m](visualizations/plot_current_experiment.m)

**Purpose:** MATLAB script to plot confusion matrices and metrics from experiments.

**Output:** Figures in `visualizations/current_experiment/`

---

# DOCKER & DEPENDENCIES

## 16. [requirements.txt](requirements.txt)

**Python Dependencies:**

```
h5py>=3.9.0              # HDF5 file I/O
numpy>=1.24.0            # Numerical computing
scikit-learn>=1.3.0      # Machine learning utilities (metrics)
pyyaml>=6.0              # YAML config parsing
tqdm>=4.65               # Progress bars
matplotlib>=3.7.0        # Plotting
seaborn>=0.12.0          # Advanced visualization
torch>=2.2.0             # PyTorch (deep learning)
```

---

## 17. [Dockerfile](Dockerfile) & [docker-compose.yml](docker-compose.yml)

**Purpose:** Containerize the training pipeline for reproducible experiments.

---

# SUMMARY: TRAINING & EVALUATION PIPELINE

## Typical Workflow:

### 1. Data Preparation

```
Run: python data/dataset.py --hdf5-path data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5 \
                            --output-path data/splits/radioml2018a_seed42_60_20_20.npz
Creates: Stratified train/val/test split indices (deterministic)
```

### 2. Training

```
Run: python -m src.train --config configs/cnn1d_quick.yaml
Loads: Config → Model (via factory) → DataLoaders → Training loop
Outputs: experiments/cnn1d_quick_YYYYMMDD_HHMMSS/
  ├── best_model.pth          (best checkpoint)
  ├── history.csv              (per-epoch metrics)
  ├── train_report.json        (training summary)
  └── config.yaml              (config copy)
```

### 3. Evaluation

```
Run: python -m src.evaluate --config experiments/cnn1d_quick_YYYYMMDD_HHMMSS/config.yaml \
                            --checkpoint experiments/cnn1d_quick_YYYYMMDD_HHMMSS/best_model.pth
Loads: Config + Checkpoint → Runs inference on test set
Outputs:
  ├── test_report.json         (overall accuracy, per-class, per-SNR)
  └── confusion_matrix.csv     (24×24 matrix)
```

---

# KEY DESIGN PATTERNS

## 1. **Configuration-Driven**

- YAML configs define all hyperparameters → CLI overrides apply at runtime
- No hardcoded magic numbers in code

## 2. **Factory Pattern**

- `build_model(config)` dispatches to correct architecture
- `build_dataloaders(config)` creates loaders from config

## 3. **Separation of Concerns**

- `train.py` — Training logic
- `evaluate.py` — Inference & metrics
- `dataset.py` — Data loading
- `utils.py` — Shared helpers
- `models/` — Architecture definitions

## 4. **Efficient Memory Usage**

- HDF5 accessed on-demand (no full 25 GB in RAM)
- Only split indices cached locally

## 5. **Reproducibility**

- Random seed control across NumPy, PyTorch, Python
- Deterministic stratified splits
- Config saved with each run

## 6. **Early Stopping**

- Monitor validation accuracy
- Save best checkpoint automatically
- Stop training if no improvement for `patience` epochs

## 7. **Comprehensive Metrics**

- Confusion matrix, per-class accuracy
- Per-SNR accuracy (important for radio signal classification)
- Precision, recall, F1 scores

---

# LAYER HIERARCHY IN MODELS

## CNN1D Layers:

```
Conv1d (64 filters)    → Feature extraction
  ↓
BatchNorm1d            → Normalization
  ↓
ReLU                   → Activation
  ↓
Conv1d (64 filters)    → Feature refinement
  ↓
BatchNorm1d → ReLU → Dropout → MaxPool1d → ... (3 conv blocks total)
  ↓
AdaptiveAvgPool1d      → Temporal aggregation
  ↓
Linear (dense_units)   → Feature mapping
  ↓
Linear (num_classes)   → Classification logits
```

## CNN-LSTM Layers:

```
Conv1d blocks          → Local feature extraction (spatial)
  ↓
LSTM                   → Temporal sequence modeling
  ↓
Mean pooling           → Temporal aggregation
  ↓
LayerNorm → Dropout → Linear → Classification logits
```

## ResNet1D Layers:

```
Conv1d (stem)          → Initial feature extraction
  ↓
BasicBlock1D × N       → Residual blocks (Layer 1, 2, 3)
  ↓                      Each with skip connections
AdaptiveAvgPool1d      → Global pooling
  ↓
Linear (num_classes)   → Classification logits
```

---

**Last Updated:** 2026-05-03
