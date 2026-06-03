---
name: kgrass
description: Use this skill when the user wants to summarize a large graph to K nodes using kGrass (spartan kGrass / kGS).
---

# kGrass Skill

## Overview

Runs **kGrass** graph summarization: compress a graph to a smaller adjacency matrix of size `K × K`.

Uses `DenseIntMapper` to remap node ids (same pattern as DPGS / kGrass.ipynb).

**Constraint:** `K` must be **less than** the number of unique nodes in the graph.

---

## Workflow

### Step 1: Understand Input

- **file**: edge list (2 columns: src, dst)
- **sep**: delimiter (`,` for CSV, `\t` for tensor files)
- **K**: summary graph size
- **strategy**: `greedy` | `linear_check` | `sample_pairs` (default)

### Step 2: Run kGrass

Small demo graph:

```bash
python my_skills/kGrass/scripts/analyze.py \
  --file tensor_data_export.csv \
  --sep "," \
  --K 400 \
  --strategy sample_pairs
```
