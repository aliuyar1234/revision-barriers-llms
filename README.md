# Revision Barriers in Language Models

Build pack for the paper:

**Revision Barriers in Language Models under Conflicting Evidence**

Canonical paper artifact in this repo:

- `paper/revision_barriers.pdf`

## Project summary
This project studies a precise empirical phenomenon:

> Two prefixes can be matched on designated incumbent/challenger state and evidence margin, yet differ in revisability because the path to that state matters. Strong "same current leader" wording is reserved for the leader-consistent subset where prefix-only canonical scoring yields the same incumbent leader for both pair members.

The paper is about measuring that difference with a matched-history behavioral construction, then reporting bounded latent and causal follow-up results without overclaiming what they show.

## Central thesis
Equal designated support state does not imply equal revisability.

## Target contribution
The target contribution is the **revision barrier** as a matched-history behavioral diagnostic, together with:
1. a matched-history behavioral construction that isolates revisability under controlled designated state,
2. a control ladder that rules out simpler readings such as `k=0` asymmetry bugs, leader inconsistency, and output-margin-only explanations,
3. a bounded latent section showing predictive prefix-boundary signal without claiming added value beyond output margin overall,
4. an honest negative causal section showing that the locked one-site steering protocol did not move the barrier,
5. a modest scope section that keeps auxiliary evidence small and honest.

## Kind of paper
- measurement-first empirical methodology paper
- behavioral paper with limited latent and causal follow-up
- not a broad benchmark
- not a generic probing paper
- not a circuit-tracing paper
- not a chain-of-thought paper

## High-level stack
- Models: Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (secondary), Qwen2.5-14B-Instruct only if the core is already alive
- Primary task: MHST (Matched-History Suspect Tracking)
- Secondary task: RC-ICL (Rule-Correction ICL), behavior-only scope check with limited weight in the final paper
- Core methods: controlled data generation, canonical option-logprob scoring, paired-valid barrier analysis, leader-consistent analysis, output-margin controls, prefix-boundary hidden-state probes, and one locked negative steering test

## Public repo contents
- `configs/`: YAML experiment configs
- `src/data/`: task generators and prompt renderers
- `src/models/`: model loading, scoring, hidden-state extraction, hooks
- `src/analysis/`: barriers, probes, stats, plots
- `scripts/`: runnable entrypoints
- `paper/`: LaTeX paper source, figures, and the tracked paper PDF

## Repo note
Internal workflow memory, experimental outputs, and project-management docs are intentionally excluded from the public repo to keep the GitHub surface lean and paper-focused.
