#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/z/codex/twist2_inspect
mkdir -p /mnt/e/codex-artifacts/teleop_feasible_training_20260917
for stem in 004 005 006 007 009; do
  path="assets/example_motions/0807_yanjie_walk_${stem}.pkl"
  git -C "$repo" restore --source=HEAD "$path"
done
git -C "$repo" restore --source=HEAD assets/ref_motions/accad_A3___Swing_t2.pkl

git -C "$repo" ls-tree -r HEAD -- \
  assets/example_motions/0807_yanjie_walk_001.pkl \
  assets/example_motions/0807_yanjie_walk_002.pkl \
  assets/example_motions/0807_yanjie_walk_003.pkl \
  assets/example_motions/0807_yanjie_walk_004.pkl \
  assets/example_motions/0807_yanjie_walk_005.pkl \
  assets/example_motions/0807_yanjie_walk_006.pkl \
  assets/example_motions/0807_yanjie_walk_007.pkl \
  assets/example_motions/0807_yanjie_walk_008.pkl \
  assets/example_motions/0807_yanjie_walk_009.pkl \
  assets/example_motions/0807_yanjie_walk_010.pkl \
  assets/ref_motions/accad_A3___Swing_t2.pkl > /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_git_ls_tree.txt

while read -r mode type blob path; do
  actual=$(git -C "$repo" hash-object "$path")
  printf '%s %s %s %s\n' "$path" "$blob" "$actual" "$([ "$blob" = "$actual" ] && echo MATCH || echo MISMATCH)"
done < /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_git_ls_tree.txt \
  | tee /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_blob_verification.txt
