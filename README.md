# FLoRG Reproduction

This repository implements a structured reproduction scaffold for **FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment**.

The code follows the supplied reproduction guide:

- FLoRG adapter uses `Delta W = L(A^T A)R` with frozen semi-orthogonal `L/R` buffers and trainable `A`.
- Server aggregation averages client Gram matrices, performs top-r eigendecomposition, then optionally applies Procrustes alignment.
- The federated loop trains clients sequentially, reloads the current global adapter at each round, averages classifier heads when enabled, logs JSONL rows, and saves adapter-only checkpoints.
- Implemented methods: `florg`, `florg_no_procrustes`, `fedit`, `federa`, `ffa_lora`.
- `fedsa_lora` and `fedex_lora` are intentionally not claimed as complete because the guide notes that their public algorithmic details need extra source verification.

## Install

```bash
pip install -r requirements.txt
```

On NVIDIA GPUs, install a CUDA PyTorch build inside the active Conda environment before running the experiments:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run A GLUE Experiment

Local GLUE TSV folders are supported through `data.root`. The provided configs point at `D:/data` and expect directories such as `D:/data/RTE`, `D:/data/WNLI`, `D:/data/QNLI`, and `D:/data/MNLI`. If a local task directory is not found, the loader falls back to Hugging Face `glue`.

The formal experiment configs follow the paper settings:

- `N = 20` clients
- full participation ratio `1.0`
- Dirichlet non-IID `rho = 0.5`
- rank `r = 4`
- LoRA/FLoRG scaling factor `16`
- local epoch `1`
- learning rate `5e-5`
- batch size `4`
- max sequence length `128`
- optimizer `AdamW`

The paper PDF does not explicitly state the total number of federated rounds; the formal configs keep `rounds: 50` from the reproduction guide.

```bash
python scripts/run_glue.py --config configs/florg_roberta_large_mrpc.yaml
```

Local RTE/WNLI/QNLI/MNLI examples:

```bash
python scripts/run_glue.py --config configs/florg_roberta_large_rte.yaml
python scripts/run_glue.py --config configs/florg_roberta_large_wnli.yaml
python scripts/run_glue.py --config configs/florg_roberta_large_qnli.yaml
python scripts/run_glue.py --config configs/florg_roberta_large_mnli.yaml
```

Faster RTE sanity run:

```bash
python scripts/run_glue.py --config configs/florg_roberta_large_rte_fast.yaml
```

Files ending with `_smoke`, `_fast`, and `_better` are convenience/debug configs and are not paper-aligned.

Procrustes ablation:

```bash
python scripts/run_glue.py --config configs/ablation_procrustes.yaml
```

FedIT baseline:

```bash
python scripts/run_glue.py --config configs/baselines.yaml --method fedit
```

FeDeRA baseline:

```bash
python scripts/run_glue.py --config configs/baselines.yaml --method federa --output-dir outputs/federa_roberta_large_mrpc
```

FFA-LoRA baseline:

```bash
python scripts/run_glue.py --config configs/baselines.yaml --method ffa_lora --output-dir outputs/ffa_lora_roberta_large_mrpc
```

## Outputs

Each run writes to `output_dir`:

- `logs/YYYYMMDD_HHMMSS.log`
- `client_stats.json`
- `checkpoints/round_XXX/global_adapter.pt`
- `checkpoints/round_XXX/metrics.json`
- `checkpoints/round_XXX/comm.json`

The JSONL-style `.log` file includes method, model, task, seed, run id, selected clients, client loss, eval metrics, communication parameters, server aggregation time, and train time.

## Collect Results

```bash
python scripts/collect_results.py --root outputs --out outputs/results/all_rounds.csv
```

Accuracy and communication table helpers live under `src/analysis`.

## Notes

GLUE validation splits are used as the evaluation proxy. Do not report these values as official GLUE test accuracy.
