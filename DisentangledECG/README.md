# DisentangledECG

Implementation of a disentangled representation learning pipeline for ECG analysis using **PTB-XL** and a **DPSOM-based** architecture.

This repository is intended for research usage: model development, controlled experiments, and reproducible evaluation.

## 1. Project structure

```text
DisentangledECG/
├── dpsom_ecg.py               # Main entry point: training + evaluation
├── dpsom_ecg_model.py         # DPSOM_ECG model definition
├── dpsom_config.py            # Experiment hyperparameters/configuration
├── ECG_Dataset.py             # PTB-XL import, preprocessing, split handling
├── ECG_Record.py              # Per-record ECG processing and beat segmentation
├── data_generator.py          # Batch and record-level data generators
├── utils.py                   # Metrics, disentanglement utilities
├── visual_utils.py            # SOM visualization utilities
├── draw_utils.py              # Signal plotting helpers
├── ptb.py                     # PTB-XL exploratory helper script
├── models/                    # Saved model checkpoints
├── data/
│   ├── PTB-XL/                # Unpacked raw PTB-XL files (not versioned)
│   └── ptbxl_100_T-12ms.pkl   # Cached processed dataset (created on first run)
└── README.md
```

## 2. Requirements

- Python 3.10+ (tested in this workspace with Python 3.13)
- `pip`
- Optional but recommended: CUDA-capable GPU for training speed

## 3. Environment setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy scikit-learn pandas wfdb neurokit2 matplotlib seaborn tqdm
```

## 4. PTB-XL dataset setup

This repository does not redistribute PTB-XL. Download it from the official source and unpack it locally.

Expected location:

```text
data/PTB-XL
```

Example:

```bash
mkdir -p data
unzip /path/to/PTB-XL.zip -d data
```

After extraction, verify that these paths exist:

- `data/PTB-XL/ptbxl_database.csv`
- `data/PTB-XL/scp_statements.csv`
- `data/PTB-XL/records100/`
- `data/PTB-XL/records500/`

## 5. Running the pipeline

### 5.1 First run (raw PTB-XL -> preprocessing -> training -> evaluation)

```bash
python dpsom_ecg.py
```

What happens on first run:

1. PTB-XL is imported from `./data/PTB-XL`.
2. Processed cache is written to `./data/ptbxl_100_T-12ms.pkl`.
3. Training is executed.
4. Evaluation and SOM visualizations are generated.

### 5.2 Subsequent runs using cached dataset

To skip re-import of raw PTB-XL, set in `dpsom_config.py`:

```python
self.use_data_cache = True
```

and ensure `data/ptbxl_100_T-12ms.pkl` exists.

### 5.3 Evaluation from a saved checkpoint

```bash
python -c "from dpsom_ecg import main; main(model_path='models/<experiment_name>/<checkpoint>.ckpt')"
```

## 6. Output artifacts

- Checkpoints: `models/<experiment_name>/<experiment_name>.ckpt`
- Cached dataset: `data/ptbxl_100_T-12ms.pkl`
- Visualizations/log artifacts: under `logs/<experiment_name>/...` (created by the training/evaluation flow)


## 7. Notes

- PTB-XL has its own license and usage policy; ensure your usage is compliant.
- Training on CPU is possible but substantially slower.
- This repository currently uses script-based configuration (no CLI parser); configuration is controlled via `dpsom_config.py`.
