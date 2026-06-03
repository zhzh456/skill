---
name: eigenpulse
description: Use this skill when the user wants to detect density surges in large streaming graphs using EigenPulse (TensorStream + sliding window).
---

# EigenPulse Skill

## Overview

Runs **EigenPulse** on streaming graph/tuple data via `spartan.TensorStream`. Input rows are typically `(user, item, time, value)` with configurable column indices.

Supports plain or **gzip** text files (`.gz`).

The skill applies compatibility fixes for `StringMapper` / `toSTensor` in spartan (required when item+time are merged into tuple keys).

---

## Workflow

### Step 1: Understand Input

Required:
- **file**: streaming tensor file (4 columns: user, item, time, count/value)

Optional parameters (defaults match `EigenPulse.ipynb`):
- **window** / **stride**: sliding window size and step (time units)
- **l** / **b**: AugSVD hyper-parameters
- **item-idx** / **ts-idx**: column indices for item and timestamp
- **col-idx**: which columns to read (default `0,1,2,3`)
- **sep**: field delimiter

Data must be **sorted or streamable by increasing time** in column `ts-idx` so sliding windows can advance.

### Step 2: Run EigenPulse

```bash
python my_skills/EigenPulse/scripts/analyze.py \
  --file ./inputData/test_beer.tensor.gz \
  --sep "," \
  --col-idx 0 1 2 3 \
  --window 20 \
  --stride 10 \
  --l 20 \
  --b 10 \
  --item-idx 1 \
  --ts-idx 2
```

Local sample (repo): `inputData/test_beer.tensor.gz` (from CFD-3 `fs1.csv`). Official tutorial data:

```bash
curl -fsSL -o ./inputData/test_beer.tensor.gz \
  https://raw.githubusercontent.com/BGT-M/spartan2-tutorials/master/inputData/test_beer.tensor.gz
```
