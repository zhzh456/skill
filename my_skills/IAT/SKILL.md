---
name: iat
description: Use this skill when the user wants to analyze inter-arrival time (IAT) patterns and suspicious micro-clusters from timestamped user event logs using spartan.IAT.
---

# IAT Skill

## Overview

Runs **IAT (Inter-Arrival Time)** analysis to detect users whose event timing forms suspicious micro-clusters (pairs of consecutive IAT values).

Pipeline:
1. Load timestamped logs via `loadTensor`
2. Map timestamps with `TimeMapper`
3. Aggregate per-user timestamp lists (`to_aggts`)
4. Compute IAT pairs via `st.IAT().calaggiat()`

Works with EagleMine-style downstream histogram analysis.

---

## Workflow

### Step 1: Understand Input

Required (one of):
- **file**: timestamped log file
- **load-aggiat**: precomputed aggiat dict (`.dictlist.gz`)

Parameters:
- **sep**: field delimiter (`\x01` for wbcovid19 logs, `,` for CSV)
- **time-col / group-col**: timestamp and user/group columns
- **time-format**: strptime format for `TimeMapper`
- **save-aggiat**: optional output path for reuse

### Step 2: Run IAT

Demo with local CFD data (integer timestamps in column 1):

```bash
python my_skills/IAT/scripts/analyze.py \
  --file ./inputData/cfd.csv \
  --sep "," \
  --time-col 1 \
  --group-col 0 \
  --time-format int \
  --top-k 10
```

Notebook-style log file (`wbcovid19_test.gz`, sep=`\x01`):

```bash
python my_skills/IAT/scripts/analyze.py \
  --file ./inputData/wbcovid19_test.gz \
  --sep $'\x01' \
  --time-col 0 \
  --group-col 1 \
  --time-format "%Y-%m-%d %H:%M:%S" \
  --save-aggiat ./output/aggiat.dictlist.gz
```

Load saved aggiat:

```bash
python my_skills/IAT/scripts/analyze.py \
  --load-aggiat ./output/aggiat.dictlist.gz
```
