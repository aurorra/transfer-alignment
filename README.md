# Transfer Alignment for Grammatical Error Detection

This repository contains the code used to study transfer between grammatical error types through representation alignment. The experiments use sentence embeddings extracted from fine-tuned multilingual BERT (mBERT) models and evaluate four alignment methods:

- Principal Component Analysis (PCA)
- Kernel PCA (KPCA)
- CORrelation ALignment (CORAL)
- Optimal Transport (OT)

The experiments transfer between three German real-word error types:

- Case errors
- Verb errors
- Capitalization errors

The repository also includes an optional hyperparameter search for KPCA and OT and a lightweight example based on precomputed embeddings.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Hardware and CUDA

The experiments reported in the paper were run on an NVIDIA RTX 3090. PyTorch and CUDA versions may need to be adapted for different GPUs.

The full experiment runner expects CUDA to be available. The lightweight `--example` mode uses precomputed embeddings and can also run without CUDA.

## Data

The synthetic real-word error datasets are available on Hugging Face:

https://huggingface.co/datasets/aurorra/synthetic-real-word-errors

For the main transfer experiments, the runner expects local data directories for the three error types:

```text
data/
├── synthetic_data_case_error_V3_100000/
├── synthetic_data_verb_V3_100000/
└── synthetic_data_cap_V3_100000/
```

Each dataset directory is expected to contain the corresponding training (`training.csv`), validation (`validation.csv`), and test (`test.csv`) data used by the experiment code.

The default hyperparameter search additionally expects a dedicated validation subset. The subset size is configured through `search_samples` in `configs/default.yaml`. With the default setting `search_samples: 2000`, the Case &rarr; Verb search expects:

```text
data/
└── synthetic_data_verb_V3_100000/
    └── subset_n=2000/
        ├── training.csv
        └── validation.csv
```

The directory name must follow the pattern `subset_n=<search_samples>`.

## Fine-Tuned mBERT Checkpoints

For a full run, the code expects one fine-tuned mBERT checkpoint per source error type. The repository includes a fine-tuning module (`transfer/src/finetuning/finetune_runner.py`) with the `run_finetuning()` function, which can be used to train the required mBERT models before running the alignment experiments.

By default, the alignment experiments expect the resulting checkpoints at:

```text
results/
├── finetuned_mbert_case_error/
│   └── baseline/init_0/bert-base-multilingual-cased-final/
├── finetuned_mbert_verb/
│   └── baseline/init_0/bert-base-multilingual-cased-final/
└── finetuned_mbert_cap/
    └── baseline/init_0/bert-base-multilingual-cased-final/
```

These checkpoints are used to extract source and target CLS embeddings.

## Quick Example

The repository contains a lightweight example based on precomputed Case &rarr; Verb embeddings.

Run:

```bash
python -m transfer.experiments.run_comparison --example
```

The example:

- uses Case &rarr; Verb
- uses seed 60
- uses the precomputed embeddings under `example/embeddings/`
- runs PCA, KPCA, CORAL, and OT
- trains the lightweight MLP classifier
- evaluates the raw transfer baseline and all alignment methods
- does not require mBERT embedding extraction

The example is intended as a functional demonstration of the pipeline. The paper results are reported as means over five random seeds and therefore should not be expected to match a single example run exactly.

Example outputs are written to:

```text
example/results/
```

## Main Experiments

Run all four alignment methods with the default settings:

```bash
python -m transfer.experiments.run_comparison
```

The default configuration uses five random seeds and evaluates all six transfer directions between Case, Verb, and Capitalization errors.

### Run Selected Alignment Methods

For example, run only KPCA and OT:

```bash
python -m transfer.experiments.run_comparison --aligners kpca ot
```

### Run Selected Seeds

```bash
python -m transfer.experiments.run_comparison --seeds 60
```

or:

```bash
python -m transfer.experiments.run_comparison --seeds 60 61 62
```

### Re-extract Embeddings

Existing embeddings are reused when their metadata match the current run.

To force new extraction:

```bash
python -m transfer.experiments.run_comparison --extract-embeddings
```

This option should be used when reproducing runtime measurements that include embedding extraction.

### Disable the Raw Transfer Baseline

```bash
python -m transfer.experiments.run_comparison --no-include-raw-baseline
```

## Hyperparameters

PCA and CORAL use the fixed settings from the experiments. KPCA and OT use the selected values from the paper by default based on a small hyperparameter search.

### Selected KPCA Parameters

| Parameter | Selected value |
|---|---:|
| `gamma` | `0.02` |
| `n_fit` | `15000` |
| `alpha` | `0.01` |
| `n_components` | `32` |
| `k_align` | `32` |

### Selected OT Parameters

| Parameter | Selected value |
|---|---:|
| `reg` (lambda) | `20.0` |
| Standardization | `True` |

## Hyperparameter Search

The repository can reproduce the minimal hyperparameter search used in the paper. The search is performed only for Case &rarr; Verb, matching the experimental setup.

### KPCA Search Space

| Parameter | Search values |
|---|---|
| `gamma` | `2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2.0, 5.0, 10.0` |
| `n_fit` | `5000, 10000, 15000` |
| `alpha` | `1e-2, 5e-2, 1e-1` |
| `n_components` | `32, 64, 128, 256` |
| `k_align` | `32, 64, 128` |

### OT Search Space

| Parameter | Search values |
|---|---|
| `reg` (lambda) | `20.0, 30.0, 50.0, 80.0, 100.0` |
| Standardization | `True, False` |

Run both searches with:

```bash
python -m transfer.experiments.run_comparison --search kpca ot
```

or run only one:

```bash
python -m transfer.experiments.run_comparison --search kpca
```

```bash
python -m transfer.experiments.run_comparison --search ot
```

During hyperparameter search, the raw transfer baseline is skipped because it does not depend on the alignment hyperparameters.

## Overriding Selected Hyperparameters

Values from `configs/default.yaml` can be overridden from the command line.

Examples:

```bash
python -m transfer.experiments.run_comparison \
    --aligners kpca \
    --kpca-gamma 0.002 \
    --kpca-n-fit 10000 \
    --kpca-alpha 0.05 \
    --kpca-n-components 64 \
    --kpca-k-align 64
```

For OT:

```bash
python -m transfer.experiments.run_comparison \
    --aligners ot \
    --ot-reg 30
```

Disable OT standardization with:

```bash
python -m transfer.experiments.run_comparison \
    --aligners ot \
    --no-ot-standardize
```

## Configuration

The default experiment settings are stored in:

```text
configs/default.yaml
```

The configuration contains:

- default alignment methods
- random seeds
- selected KPCA settings
- selected OT settings
- KPCA search space
- OT search space
- example settings

A different configuration file can be supplied with:

```bash
python -m transfer.experiments.run_comparison --config path/to/config.yaml
```

Command-line arguments override the selected values loaded from the configuration.

## Outputs

The experiments write results to method- and transfer-specific output directories.

Depending on the run, outputs include:

- extracted embeddings
- aligned embeddings
- trained MLP classifiers
- prediction metrics
- hyperparameter-search results
- timing information
- CodeCarbon energy and emissions measurements

The main results are stored in JSONL summaries for each seed and transfer direction.

## Reproducibility

The code fixes Python, NumPy, and PyTorch random seeds and enables deterministic PyTorch behavior where supported.

For runtime comparisons, embedding extraction should be enabled because reusing cached embeddings measures loading time rather than full embedding-extraction time.

The exact package versions used on the original experiment machine should be reproduced through `requirements.txt`. CUDA/PyTorch builds may still need to be adapted to the available GPU hardware.

## Citation

The core embedding-alignment approach was published at NLDB 2026:

Masanti, C., Witschel, H.-F., & Riesen, K. (2026). *Efficient Error-Type Transfer for Grammatical Error Detection via Embedding Alignment*. NLDB 2026, LNCS 16696, 189–203. https://doi.org/10.1007/978-3-032-29532-3_14

An extended journal version including KPCA and Optimal Transport is currently under review.
