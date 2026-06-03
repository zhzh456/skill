---
name: dpgs
description: Use this skill when the user wants to run the DPGS graph diffusion / propagation algorithm on edge-list graph data. Supports CSV or TSV-style edge lists with optional parameters. Designed for large sparse graph adjacency reconstruction and propagation analysis.
---

# DPGS Skill

## Overview

Runs the **DPGS (Diffusion-based Propagation Graph Sampling)** algorithm using the `spartan` library.

This skill processes graph edge-list data and builds a dense-indexed tensor representation before applying DPGS propagation.

Supported input formats:
- Edge-list text files (CSV / TSV)
- Files without headers

---

## Workflow

### Step 1: Understand Input Requirements

Identify from user input:
- **File path(s)**: one or more graph datasets
- **Separator (`sep`)**: default is `\t`
- **hasvalue**: whether edges include weights (default: false)
- **T**: number of propagation iterations (default: 1)

---

## Step 2: Run DPGS

### Example Usage

#### TSV / Edge List Input (default)

```bash
python my_skills/DPGS/scripts/analyze.py \
  --files soc-Epinions1.csv \
  --sep "," \
  --hasvalue false \
  --T 5
