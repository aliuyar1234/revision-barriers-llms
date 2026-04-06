# Revision Barriers in Language Models under Conflicting Evidence

[![Paper PDF](https://img.shields.io/badge/Paper-Download%20PDF-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](paper/revision_barriers.pdf?raw=1)
[![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-008080?style=for-the-badge&logo=latex&logoColor=white)](paper/)
[![Curated Results](https://img.shields.io/badge/Results-Curated%20Artifacts-1F6FEB?style=for-the-badge)](outputs/public/)
[![Research Type](https://img.shields.io/badge/Research-Behavioral%20Methodology-5B21B6?style=for-the-badge)](#overview)

**Ali Uyar**  
Independent Researcher

## Overview

This repository accompanies the paper **Revision Barriers in Language Models under Conflicting Evidence**.

The project studies a simple but under-measured question: when two model states look the same under current preference summaries, are they equally revisable once later conflicting evidence arrives?

The central empirical answer here is **no**.

Using a matched-history construction, the paper shows that two prefixes can be matched on designated incumbent/challenger state and evidence margin, yet differ systematically in how much later challenger evidence is required to reverse that state. The resulting quantity is the **revision barrier**.

The paper is intentionally narrow in its claims:

- it is a **behavioral methodology paper**
- the main contribution is a clean matched-history diagnostic of revisability
- latent probing is reported as a bounded follow-up
- causal steering is reported as an honest negative result
- scope evidence is limited and kept secondary

## Main Results

### 1. Strong behavioral effect on MHST/Qwen

On held-out paired-valid test pairs, committed histories are substantially harder to reverse than fresh histories:

- mean paired barrier gap `ΔB_pair^cap = 2.7784`
- 95% CI `[2.3836, 3.1893]`
- leader-consistent subset remains same-sign at `1.8496`
- the required `k = 0` cell is exactly null
- the output-margin control remains same-sign

### 2. Current-state summaries are insufficient

A simple state-only predictor built from designated margin and prefix-only output margin still underpredicts the paired history effect:

- actual paired mean-flipability gap `= 0.3969`
- predicted gap `= 0.3531`
- residual gap `= 0.0439`
- 95% CI `[0.0131, 0.0756]`

This is the paper's second main result and the clearest compact statement of the core claim: **state matching is insufficient for revisability**.

### 3. Internal limits are informative but secondary

- the prefix-boundary probe is predictive beyond designated margin `m`
- it does **not** beat the prefix-only output-margin baseline overall
- the locked one-site steering protocol does **not** produce a qualifying barrier-moving effect

These sections increase trust in the paper by bounding interpretation rather than carrying the main claim.

### 4. Scope evidence is limited

- Llama MHST is same-sign, but auxiliary
- RC-ICL is same-sign on its main summary, but fails its cleanliness threshold and is therefore not treated as a clean second success

## Paper Artifact

Canonical public paper artifact:

- [paper/revision_barriers.pdf](paper/revision_barriers.pdf?raw=1)

LaTeX source:

- [paper/](paper/)

## Public Artifact Map

This public repository is intentionally lean. It tracks the paper, source code, configs, and a curated subset of final result artifacts.

### Tracked

- `paper/`
  - LaTeX manuscript source
  - paper figures
  - tracked PDF
- `src/`
  - data generation, scoring, analysis, probing, and intervention code
- `scripts/`
  - runnable experiment and paper-asset entrypoints
- `configs/`
  - YAML experiment configurations
- `outputs/public/`
  - curated final summaries directly supporting the paper

### Intentionally excluded

The public repo does **not** track:

- internal workflow state
- project-management memory
- large raw output trees
- local caches and build products
- staging bundles and scratch artifacts

That exclusion is deliberate: the goal is a clean, paper-facing research repository rather than a dump of internal execution history.

## Curated Public Results

The directory [outputs/public/](outputs/public/) contains the small subset of final quantitative artifacts most useful for inspection on GitHub:

- `behavior/`
  - main MHST/Qwen summary
  - Llama MHST auxiliary summary
  - RC-ICL auxiliary summary
  - MHST/Qwen companion mean-flip summary
- `analysis/`
  - completed state-only insufficiency decomposition
- `latent/`
  - original and rerun latent baseline-comparison summaries
- `causal/`
  - original and rerun causal summary artifacts

## Repository Structure

```text
configs/        Experiment configurations
paper/          LaTeX paper source, figures, tracked PDF
scripts/        Experiment and paper-asset entrypoints
src/            Data, model, and analysis code
outputs/public/ Curated final result artifacts for public inspection
```

## Build

To rebuild the paper locally:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Positioning

This project should be read as:

- a measurement-first empirical methodology paper
- a paper about **history-sensitive revisability**
- not a broad benchmark
- not a generic probing paper
- not a circuit-tracing paper
- not a claim that a latent or causal mechanism has been established

## Why this repo exists

The aim of this repository is to make the paper inspectable at three levels:

1. the final paper artifact
2. the code/config surface used to produce the scientific claims
3. a curated public slice of the final quantitative outputs

That combination is enough to let a reader understand the paper as a scientific object without burying the repository under internal workflow debris.
