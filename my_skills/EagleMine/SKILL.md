---
name: eaglemine
description: Use this skill when the user wants vision-guided micro-cluster anomaly detection with EagleMine on histogram data or by building a histogram from a bipartite graph.
---

# EagleMine Skill

## Overview

Runs **EagleMine** for micro-cluster detection on 2D histograms. Two input modes:

1. **histogram** — use pre-built `histogram.out`, `node2hcel.out`, `hcel2avgfeat.out` (from `eaglemine_data.zip` or prior runs)
2. **graph** — load an edge-list tensor, extract features (e.g. `outdegree2hubness`), build a histogram, then run EagleMine

**Dependency:** spartan's EagleMine requires legacy **pomegranate 0.14.x** (not the PyTorch 1.x package). In conda env `AdvPEFT`:

```bash
pip install 'pomegranate==0.14.9' -i https://pypi.tuna.tsinghua.edu.cn/simple
# or: -i https://mirrors.aliyun.com/pypi/simple/
```

---

## Workflow

### Step 1: Understand Input

**Histogram mode** — paths to three `.out` files.

**Graph mode** — graph tensor path plus feature type and bin settings.

Run parameters: `waterlevel_step`, `prune_alpha`, `min_pts`, `strictness`.

### Step 2: Run EagleMine

Histogram files (after extracting `inputData/eaglemine_data.zip`):

```bash
python my_skills/EagleMine/scripts/analyze.py \
  --input-mode histogram \
  --histogram ./inputData/histogram.out \
  --node2hcel ./inputData/node2hcel.out \
  --hcel2avgfeat ./inputData/hcel2avgfeat.out \
  --out-dir ./output \
  --voctype dtmnorm \
  --mode 2 \
  --mix-comps 2
```

Build histogram from graph (example uses repo edge list):

```bash
python my_skills/EagleMine/scripts/analyze.py \
  --input-mode graph \
  --graph-file soc-Epinions1.csv \
  --sep "," \
  --hasvalue false \
  --feature-type outdegree2hubness \
  --n-bins 80 \
  --out-dir ./output
```
