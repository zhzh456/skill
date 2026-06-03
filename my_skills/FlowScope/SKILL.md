---
name: flowscope
description: Use this skill when the user wants to detect money-laundering style dense flows using FlowScope on two coupled bipartite transaction graphs (in-flow and out-flow matrices).
---

# FlowScope Skill

## Overview

Runs **FlowScope** on two sparse bipartite graphs that share a middle dimension (e.g., account → merchant, merchant → counterparty).

Each input file must be a **3-column** edge list: `source, destination, amount`.

If your data has **4 columns** (e.g. CFD `fs1.csv` with an extra time/mode column), pass `--four-dim` to project columns `0,1,last` automatically, or use the pre-generated `*.flowscope_tmp.csv` files under `inputData/CFD-3/`.

---

## Workflow

### Step 1: Understand Input

Required:
- **fs1-file**: in-flow bipartite edges (src, mid, value)
- **fs2-file**: out-flow bipartite edges (mid, dst, value)

Parameters:
- **k**: number of suspicious flow blocks
- **alpha**: imbalance weight (paper λ, default `4`)
- **maxsize**: optional `(n1, n2, n3)` cap per partite set

### Step 2: Run FlowScope

Using CFD-3 prepared 3-column tensors:

```bash
python my_skills/FlowScope/scripts/analyze.py \
  --fs1-file ./inputData/CFD-3/fs1.csv.flowscope_tmp.csv \
  --fs2-file ./inputData/CFD-3/fs2.csv.flowscope_tmp.csv \
  --k 1 \
  --alpha 4 \
  --maxsize 10 10 10
```

From raw 4-column CFD files:

```bash
python my_skills/FlowScope/scripts/analyze.py \
  --fs1-file ./inputData/CFD-3/fs1.csv \
  --fs2-file ./inputData/CFD-3/fs2.csv \
  --four-dim \
  --k 3 \
  --alpha 4 \
  --maxsize 10 10 10
```
