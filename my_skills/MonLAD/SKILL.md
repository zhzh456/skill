---
name: monlad
description: Use this skill when the user wants to detect money laundering agent accounts in transaction streams using MonLAD (ZeroOutCore / CFD format).
---

# MonLAD Skill

## Overview

Runs **MonLAD** (Money Laundering Agents Detection) on streaming transaction data.

Two input formats (`--has-edge`):
- **true**: `(source_id, dest_id, timestamp, weight)`
- **false** (CFD demo): `(account_id, timestamp, transaction_type, weight)` with types like `PRIJEM` (in) / `VYDAJ` (out)

Bundled MonLAD code is used when `st.MonLAD` is unavailable in the installed spartan2 version.

---

## Workflow

### Step 1: Understand Input

Required:
- **file**: transaction stream CSV

Key parameters:
- **delta-up / delta-down / epsilon**: fan-in/fan-out thresholds (default `10000`)
- **window / stride / ts-idx**: sliding window over timestamp column
- **source-type / des-type**: transaction type labels for CFD format

### Step 2: Run MonLAD

CFD demo data (`has_edge=false`):

```bash
python my_skills/MonLAD/scripts/analyze.py \
  --file ./inputData/cfd.csv \
  --sep "," \
  --col-idx 0 1 2 3 \
  --col-types int,int,str,float \
  --has-edge false \
  --delta-up 10000 \
  --delta-down 10000 \
  --epsilon 10000 \
  --window 1 \
  --stride 1 \
  --ts-idx 1 \
  --source-type VYDAJ \
  --des-type PRIJEM
```

With anomaly scoring:

```bash
python my_skills/MonLAD/scripts/analyze.py \
  --file ./inputData/cfd.csv \
  --has-edge false \
  --run-anomaly \
  --alpha 0.5 \
  --k 1 \
  --p 0.8
```
