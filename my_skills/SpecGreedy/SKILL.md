---
name: specgreedy
description: Use this skill when the user wants to find multiple dense subgraphs on bipartite edge-list graph data using SpecGreedy (spartan.Specgreedy).
---

# SpecGreedy Skill

## Overview

Runs **SpecGreedy** via `st.Specgreedy` to iteratively find `T` dense subgraphs on a bipartite graph.

Supported input:
- Edge-list CSV / ZIP (2 columns, space- or comma-separated)
- PKL tensor

Note: the spartan API class name is **`Specgreedy`** (lowercase g).

---

## Workflow

### Step 1: Understand Input

- **file**: edge list path
- **sep**: delimiter (default space, as in `plain_graph_small.zip`)
- **T**: number of dense blocks to extract (default `5`)
- **bipartite**: treat graph as bipartite (default `true`)

### Step 2: Run SpecGreedy

```bash
python my_skills/SpecGreedy/scripts/analyze.py \
  --file tensor_data_export.csv \
  --sep "," \
  --header none \
  --hasvalue false \
  --bipartite true \
  --T 5
```
