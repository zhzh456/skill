---
name: cubeflow
description: Use this skill when the user wants to detect suspicious transaction patterns (e.g., money laundering) using the CubeFlow algorithm on coupled tensor data.
---

# CubeFlow Skill

## Overview

Runs the **CubeFlow** algorithm for fraud and money laundering detection.

CubeFlow models transaction data as **coupled tensors** and detects suspicious dense blocks that correspond to coordinated transfer behaviors.

This skill is designed for:
- Financial transaction analysis
- Fraud detection
- Dense subgraph / block discovery

---

## Workflow

### Step 1: Understand Input

Required:
- **amt-file**: transaction tensor (e.g., account → merchant)
- **cmt-file**: coupled tensor (e.g., merchant → counterparty)

Optional:
- **gt-file**: ground truth for evaluation

Parameters:
- **alpha**: imbalance cost weight (0~1)
- **k**: number of suspicious blocks
- **dim**: tensor dimension (3 or 4)

---

## Step 2: Run CubeFlow

### Example

```bash
python my_skills/CubeFlow/scripts/analyze.py \
  --amt-file ./inputData/CFD-3/fs1.csv \
  --cmt-file ./inputData/CFD-3/fs2.csv \
  --gt-file ./inputData/CFD-3/gt.npy \
  --alpha 0.8 \
  --k 1 \
  --dim 3