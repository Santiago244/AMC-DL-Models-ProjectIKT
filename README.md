# AMC Thesis — Dataset Overview, Model Structures & Experiment Tracking

# Dataset: RadioML 2018.01A (DeepSig)

## 2.4 Dataset Overview — RadioML 2018.01A

### What is RadioML 2018.01A?

RadioML 2018.01A is the larger, more challenging successor to 2016.10a, released by DeepSig.
It is the current standard benchmark for AMC research. The dataset contains complex I/Q samples
of simulated radio signals generated with realistic channel impairments including AWGN,
multipath fading, sample rate offset, carrier frequency offset, and phase noise.

Download: https://www.deepsig.ai/datasets
File: GOLD_XYZ_OSC.0001_1024.hdf5 (~25 GB)

### Structure of the Dataset

| Property              | Value                                      |
| --------------------- | ------------------------------------------ |
| Format                | HDF5 (.hdf5)                               |
| Signal representation | Complex I/Q samples                        |
| Sample length         | 1024 complex samples → stored as (1024, 2) |
| SNR range             | -20 dB to +30 dB (step 2 dB) → 26 levels   |
| Modulation classes    | 24                                         |
| Samples per class/SNR | 4,096                                      |
| Total samples         | 24 × 26 × 4096 = 2,555,904                 |
| Total size            | ~25 GB uncompressed                        |

---

### Modulation Classes (24 total)

| Category    | Modulations                                |
| ----------- | ------------------------------------------ |
| Analog AM   | AM-DSB-WC, AM-DSB-SC, AM-SSB-WC, AM-SSB-SC |
| Analog FM   | FM, PM                                     |
| Digital PSK | BPSK, QPSK, 8PSK, 16PSK, 32PSK             |
| Digital QAM | 16QAM, 32QAM, 64QAM, 128QAM, 256QAM        |
| Digital ASK | OOK, 4ASK, 8ASK                            |
| Digital FSK | GMSK, OQPSK                                |
| Other       | 16APSK, 32APSK, DQPSK                      |

---

### HDF5 File Structure

The HDF5 file contains three arrays:

| Key | Shape              | Description                       |
| --- | ------------------ | --------------------------------- |
| X   | (2555904, 1024, 2) | I/Q samples — axis 2 is [I, Q]    |
| Y   | (2555904, 24)      | One-hot encoded modulation labels |
| Z   | (2555904,)         | SNR value in dB for each sample   |

---

### Loading the Dataset

```python
import h5py
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────
f = h5py.File("GOLD_XYZ_OSC.0001_1024.hdf5", "r")

X = f["X"][:]   # shape: (2555904, 1024, 2)  — float32
Y = f["Y"][:]   # shape: (2555904, 24)        — one-hot float32
Z = f["Z"][:]   # shape: (2555904,)           — float64 SNR in dB

f.close()

# ── Convert labels ────────────────────────────────────────────────────
y = np.argmax(Y, axis=1)   # integer class labels, shape: (2555904,)

# ── Class names (ordered as in dataset) ──────────────────────────────
CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
    "AM-SSB-WC", "AM-SSB-SC", "DQPSK", "PM"
]

# ── Reshape X for PyTorch: (N, 2, 1024) ──────────────────────────────
# PyTorch Conv1d expects (batch, channels, length)
X = X.transpose(0, 2, 1).astype(np.float32)  # → (2555904, 2, 1024)

print(f"X shape : {X.shape}")   # (2555904, 2, 1024)
print(f"y shape : {y.shape}")   # (2555904,)
print(f"Z range : {Z.min()} to {Z.max()} dB")
print(f"Classes : {len(CLASSES)}")
```

### Train / Val / Test Split

Split by index (not random shuffle of the full array — slow on 25 GB):

```python
from sklearn.model_selection import train_test_split

indices = np.arange(len(X))

# Stratify by modulation + SNR so every split has the same class/SNR balance.
snr_levels, snr_codes = np.unique(Z, return_inverse=True)
stratify_key = y * len(snr_levels) + snr_codes

idx_train_val, idx_test = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=stratify_key
)
idx_train, idx_val = train_test_split(
    idx_train_val,
    test_size=0.25,  # 25% of the remaining 80% = 20% of the full dataset
    random_state=42,
    stratify=stratify_key[idx_train_val],
)

# Result:
# Train: ~1,533,542 samples
# Val:   ~511,181  samples
# Test:  ~511,181  samples
```

### Filtering by SNR — Core of Your Thesis

Evaluate each model at every SNR level separately:

```python
def get_snr_subset(X, y, Z, snr_value):
    mask = Z == snr_value
    return X[mask], y[mask]

snr_levels = np.unique(Z)  # [-20, -18, ..., 28, 30]

for snr in snr_levels:
    X_snr, y_snr = get_snr_subset(X_test, y_test, Z_test, snr)
    acc = evaluate(model, X_snr, y_snr)
    print(f"SNR {snr:+3d} dB  →  Accuracy: {acc:.3f}")
```

---

### Memory-Efficient PyTorch Dataset (recommended for 25 GB file)

```python
import torch
from torch.utils.data import Dataset
import h5py

class RadioMLDataset(Dataset):
    def __init__(self, hdf5_path, indices):
        self.path    = hdf5_path
        self.indices = indices
        self.file    = None      # opened lazily per worker

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, "r")

        i = self.indices[idx]
        x   = self.file["X"][i].T.astype(np.float32)   # (2, 1024)
        y   = int(np.argmax(self.file["Y"][i]))
        snr = float(self.file["Z"][i])

        return torch.tensor(x), torch.tensor(y, dtype=torch.long), snr
```

---

## Model Structure

CNN Interface:

- Input: (batch, 2, 1024) — I and Q channels, 1024 samples
- Output: (batch, 24) — logits for 24 modulation classes

---

### Model: CNN (Deep Learning)

Based on VT-CNN2 (O'Shea & Hoydis 2017), adapted for 1024-sample inputs and 24 classes.

```
Input: (batch, 2, 1024)
  ↓
Conv1d(2  → 64,  kernel=3, pad=1) + ReLU + Dropout(p)
Conv1d(64 → 64,  kernel=3, pad=1) + ReLU + Dropout(p)
MaxPool1d(2)                             → (batch, 64, 512)
  ↓
Conv1d(64 → 128, kernel=3, pad=1) + ReLU + Dropout(p)
MaxPool1d(2)                             → (batch, 128, 256)
  ↓
Flatten()                                → (batch, 32768)
  ↓
Linear(32768 → 256) + ReLU + Dropout(p)
Linear(256 → 24)
  ↓
Output logits: (batch, 24)
```

```python
# src/models/cnn.py
import torch.nn as nn

class CNN_AMC(nn.Module):
    def __init__(self, num_classes=24, filters=64, kernel_size=3,
                 dropout=0.5, dense_units=256):
        super().__init__()
        pad = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(2,        filters,   kernel_size, padding=pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(filters,  filters,   kernel_size, padding=pad), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),                   # 1024 → 512

            nn.Conv1d(filters,  filters*2, kernel_size, padding=pad), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),                   # 512 → 256
        )

        flat_dim = filters * 2 * 256           # 128 * 256 = 32768

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, dense_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Hyperparameters to track:

- filters: 32, 64, 128
- kernel_size: 3, 5, 7
- dropout: 0.3, 0.5, 0.6
- dense_units: 128, 256, 512
- learning_rate: 0.001, 0.0005, 0.0001
- batch_size: 256, 512, 1024
- optimizer: adam, adamw
- weight_decay: 0, 1e-4

---

## Project Structure

```
amc-thesis/
├── data/
│   └── .gitkeep                     # directory tracked; dataset file is NOT
├── src/
│   ├── dataset.py                   # HDF5 loading, Dataset class, SNR splits
│   ├── train.py                     # Training loop — reads YAML config
│   ├── evaluate.py                  # Per-SNR accuracy + confusion matrix
│   └── models/
│       ├── __init__.py
│       └── cnn.py
├── configs/
│   ├── cnn_baseline.yaml
│   ├── cnn_large_filters.yaml
│   └── cnn_low_dropout.yaml
├── experiments/                     # Auto-generated — gitignored
│   └── run_20240401_143022/
│       ├── config.yaml              # copy of config used for this run
│       ├── metrics.json             # per-SNR accuracy results
│       └── confusion_matrix.png
├── notebooks/
│   └── exploration.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Config File Format

Each experiment is fully defined by a YAML config. No hardcoded hyperparameters.

```yaml
# configs/cnn_baseline.yaml

experiment_name: "cnn_baseline"

model:
  type: "cnn"
  filters: 64
  kernel_size: 3
  dropout: 0.5
  dense_units: 256

training:
  batch_size: 512
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0
  early_stopping_patience: 10

data:
  hdf5_path: "/data/GOLD_XYZ_OSC.0001_1024.hdf5"
  train_split: 0.6
  val_split: 0.2
  test_split: 0.2
  random_seed: 42

evaluation:
  per_snr: true
```

---

## Docker Setup

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

CMD ["python", "src/train.py", "--config", "configs/cnn_baseline.yaml"]
```

### requirements.txt

```
torch>=2.2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
h5py>=3.9
pyyaml>=6.0
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
tqdm>=4.65
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  train:
    build: .
    volumes:
      - ./data:/data:ro # dataset mounted read-only
      - ./experiments:/workspace/experiments # results written out
      - ./configs:/workspace/configs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    command: >
      python src/train.py
      --config configs/cnn_baseline.yaml
```

Run any experiment:

```bash
docker-compose run train python src/train.py --config configs/cnn_large_filters.yaml
```

Run CNN-LSTM CUDA

```bash
python -m src.train --config configs/cnn_lstm_quick.yaml --device cuda
```

ResNet1D (CUDA):

```bash
python -m src.train --config configs/resnet1d_quick.yaml --device cuda
```

Example on how to override batch size or epochs at runtime

```bash
python -m src.train --config configs/cnn_lstm_quick.yaml --device cuda --batch-size 128 --epochs 100
python -m src.train --config configs/resnet1d_quick.yaml --device cuda --batch-size 128 --epochs 100
```

## Git + GitHub Workflow for Hyperparameter Tracking

### .gitignore

```gitignore
# Dataset — never commit
data/*.hdf5
data/*.pkl
data/*.tar.gz

# Large generated outputs
experiments/
*.pth           # model weights

# Python
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
```

### Branching Strategy

```
main                               ← stable, working code only
  ├── experiment/cnn-baseline
  ├── experiment/cnn-larger-filters
  └── experiment/cnn-low-dropout
```

### Workflow for Each Experiment

```bash
# 1. Branch off main
git checkout main
git checkout -b experiment/cnn-low-dropout

# 2. Edit the config — this IS your change record
nano configs/cnn_baseline.yaml    # dropout: 0.5 → 0.3

# 3. Commit the config change
git add configs/cnn_baseline.yaml
git commit -m "experiment: reduce CNN dropout 0.5 → 0.3"

# 4. Run inside Docker
docker-compose run train python src/train.py --config configs/cnn_baseline.yaml

# 5. Commit the result summary (not weights)
git add experiments/run_*/metrics.json experiments/run_*/confusion_matrix.png
git commit -m "results: CNN dropout=0.3 — peak acc improves +1.1% at +10dB"

# 6. Push and open a PR
git push origin experiment/cnn-low-dropout
```

### What to Commit vs. Not

| Artifact              | Commit? | Reason                              |
| --------------------- | ------- | ----------------------------------- |
| Config YAML files     | YES     | Fully defines the experiment        |
| Source code changes   | YES     | Small, essential                    |
| metrics.json per run  | YES     | Small, human-readable results       |
| Confusion matrix PNGs | YES     | Visual record                       |
| .hdf5 dataset         | NO      | 25 GB — keep in cloud/local storage |
| .pth model weights    | NO      | Large — store separately if needed  |
| Full experiment logs  | NO      | Too verbose                         |

### GitHub Pull Request as Experiment Record

Use the PR description as your experiment log:

```markdown
## Experiment: CNN Dropout = 0.3

**Change:** dropout 0.5 → 0.3

**Hypothesis:** Model may be underfitting at high SNR with heavy dropout.

**Results:**

| SNR    | Baseline (0.5) | This run (0.3) |
| ------ | -------------- | -------------- |
| -10 dB | 34.2%          | 33.8%          |
| 0 dB   | 72.1%          | 73.4%          |
| +10 dB | 89.3%          | 90.1%          |
| +18 dB | 92.1%          | 93.2%          |

**Conclusion:** Slight improvement at high SNR. Merge as new baseline.
```

---

## Evaluation Protocol

Every model must be evaluated identically to allow fair comparison in Chapter 4.

```python
# src/evaluate.py
import torch, json
import numpy as np

def evaluate_per_snr(model, X_test, y_test, Z_test, snr_levels, device):
    model.eval()
    results = {}

    for snr in snr_levels:
        mask  = Z_test == snr
        X_snr = torch.tensor(X_test[mask]).to(device)
        y_snr = torch.tensor(y_test[mask]).to(device)

        with torch.no_grad():
            preds = model(X_snr).argmax(dim=1)
            acc   = (preds == y_snr).float().mean().item()

        results[int(snr)] = round(acc * 100, 2)

    return results  # e.g. {-20: 4.2, -18: 5.1, ..., 30: 95.1}


def save_metrics(results, config_name, output_path):
    payload = {
        "model": config_name,
        "per_snr_accuracy": results,
        "overall_accuracy": round(np.mean(list(results.values())), 2)
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
```

---

## Quick Start Checklist

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/amc-thesis.git
cd amc-thesis

# 2. Place dataset
mkdir -p data
mv ~/Downloads/GOLD_XYZ_OSC.0001_1024.hdf5 data/

# 3. Build Docker image
docker build -t amc-thesis .

# 4. Run CNN baseline (GPU recommended)
docker-compose run train python src/train.py --config configs/cnn_baseline.yaml

# 6. Commit results
git add configs/ experiments/*/metrics.json
git commit -m "results: CNN baseline complete"
git push
```

---

## Current Automated PyTorch Experiment Flow

The project now uses the saved split file:

```text
data/splits/radioml2018a_seed42_60_20_20.npz
```

and the raw dataset:

```text
data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5
```

Run experiments locally:

```bash
python -m src.train --config configs/cnn_baseline.yaml
python -m src.train --config configs/cnn_lstm_baseline.yaml
python -m src.train --config configs/resnet1d_baseline.yaml
```

Run experiments with Docker:

```bash
docker compose build
docker compose run --rm train_cnn1d
docker compose run --rm train_cnn_lstm
docker compose run --rm train_resnet1d
```

Each run writes a timestamped folder in:

```text
experiments/
```

Expected outputs:

```text
config.yaml
train_report.json
history.csv
best_model.pth
```

After training, evaluate a checkpoint:

```bash
python -m src.evaluate \
  --config configs/cnn_baseline.yaml \
  --checkpoint experiments/<run_folder>/best_model.pth
```

Evaluation writes:

```text
test_report.json
confusion_matrix.csv
```

---

## Local NVIDIA/CUDA Setup (Verified)

Environment verified on Windows (May 2026):

```text
NVIDIA-SMI 572.83
Driver Version: 572.83
CUDA Version: 12.8
```

Create and activate a local virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install PyTorch with CUDA 12.8 wheels using pip:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is available from PyTorch:

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'PyTorch Version: {torch.__version__}')"
```

Expected output:

```text
CUDA Available: True
CUDA Version: 12.8
PyTorch Version: 2.11.0+cu128
```
