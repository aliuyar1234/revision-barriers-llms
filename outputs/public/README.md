# Public Results

This directory contains a curated public subset of final result artifacts.

Purpose:

- make the key paper-backed quantitative outputs visible on GitHub
- avoid tracking the full internal `outputs/` tree, which contains workflow state, staging bundles, logs, and other generated artifacts that are not useful in the public repo

Included here:

- `behavior/`
  - main MHST/Qwen summary
  - Llama MHST auxiliary summary
  - RC-ICL auxiliary summary
  - MHST/Qwen companion mean-flipability summary
- `analysis/`
  - completed `M8` state-only insufficiency decomposition
- `latent/`
  - original and rerun latent baseline-comparison summaries
- `causal/`
  - original and rerun causal summary artifacts

These files are intended to be:

- easy to inspect
- sufficient to understand the main supported quantitative claims
- small enough to keep the public repo lean

The full internal `outputs/` tree remains intentionally untracked.
