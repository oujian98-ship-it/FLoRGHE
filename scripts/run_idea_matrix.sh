#!/usr/bin/env bash
set -euo pipefail

configs=(
  idea_update_space_roberta_large_rte
  idea_no_procrustes_roberta_large_rte
  idea_prefix_safe_roberta_large_rte
  idea_update_space_roberta_large_mrpc
  idea_prefix_safe_roberta_large_mrpc
  idea_no_procrustes_roberta_large_mrpc
  idea_update_space_roberta_large_qnli
  idea_prefix_safe_roberta_large_qnli
  idea_no_procrustes_roberta_large_qnli
  idea_update_space_roberta_large_wnli
  idea_prefix_safe_roberta_large_wnli
  idea_no_procrustes_roberta_large_wnli
  idea_gamma0_roberta_large_rte
  idea_gamma05_roberta_large_rte
  idea_gamma1_roberta_large_rte
  idea_homogeneous_r4_roberta_large_rte
  idea_mild_hetero_roberta_large_rte
  idea_strong_hetero_roberta_large_rte
  idea_sensitivity_roberta_large_rte
  idea_rho1_roberta_large_rte
  idea_rho05_roberta_large_rte
  idea_rho01_roberta_large_rte
  idea_participation1_roberta_large_rte
  idea_participation05_roberta_large_rte
  idea_participation02_roberta_large_rte
)

for cfg in "${configs[@]}"; do
  python scripts/run_glue_idea.py --config "configs/${cfg}.yaml"
done
