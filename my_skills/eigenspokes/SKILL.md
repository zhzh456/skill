---
name: eigenspokes
description: Use this skill when the user wants to run the Eigenspokes graph processing algorithm on edge-list style graph data. Supports CSV with configurable delimiter and PKL input.
---

# Eigenspokes Skill

## Overview

Runs the Eigenspokes graph processing algorithm using the `spartan` library.

This skill is designed for graph data stored as edge lists. It supports:
- **CSV** input with configurable delimiter
- **PKL** input

The input graph is converted to an `STensor`, then processed with `st.Eigenspokes`.

## Workflow

### Step 1: Understand Requirements

Identify:
- **File location**: Path(s) to uploaded files
- **File format**: CSV or PKL
- **Separator** (optional): For CSV files, such as `,`, `\t`, or space
- **Header** (optional): Whether the CSV has a header row
- **Has value** (optional): Whether the graph edges include a value column

### Step 2: Run Analysis

#### CSV Input
```bash
python my_skills/eigenspokes/scripts/analyze.py \
  --files tensor_data_export.csv \
  --sep "," \
  --header none \
  --hasvalue false