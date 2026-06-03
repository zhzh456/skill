---
name: holoscope
description: Use this skill when the user wants to run HoloScope topology-and-spike aware fraud detection on bipartite graph edge-list data. Supports CSV with optional temporal column and TimeMapper.
---

# HoloScope Skill

## Overview

Runs **HoloScope** using the `spartan` library. HoloScope detects dense suspicious blocks on bipartite graphs using contrast suspiciousness (topology level 0) and optional holistic signals.

Supported input:
- **CSV / edge-list** files (2 or 3+ columns)
- Optional **time column** with `TimeMapper` when `hasvalue=true` and `--time-col` is set

**Note:** Newer NumPy (≥1.24) can break HoloScope inside spartan; this skill applies a small compatibility patch before calling the model.

---

## Workflow

### Step 1: Understand Input

Identify:
- **File path**: graph edge list
- **sep**: delimiter (default `,`)
- **header**: `none` or `infer`
- **hasvalue**: whether edges have weights / extra attribute columns
- **time-col**: column index for dates (only when using temporal mode)
- **level**: `0` = topology only; higher levels add holistic signals
- **k**: number of suspicious blocks to return

### Step 2: Run HoloScope

```bash
python my_skills/HoloScope/scripts/analyze.py \
  --file tensor_data_export.csv \
  --sep "," \
  --header none \
  --hasvalue false \
  --level 0 \
  --k 1
```
