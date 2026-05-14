#!/usr/bin/env bash
set -euo pipefail

for method in florg florg_no_procrustes fedit federa ffa_lora; do
  for task in mrpc qqp mnli qnli wnli rte; do
    config="configs/${method}_roberta_large_${task}.yaml"
    if [[ -f "$config" ]]; then
      python scripts/run_glue.py --config "$config"
    else
      echo "skip missing $config"
    fi
  done
done
