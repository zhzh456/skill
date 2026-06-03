---
name: fraudar
description: Use this skill when the user wants to run the Fraudar graph anomaly detection algorithm on edge-list graph data. Supports CSV with configurable delimiter and PKL input.
---

# Fraudar Skill

## Overview

Runs the Fraudar graph anomaly detection algorithm using the `spartan` library.

This skill is designed for graph data stored as edge lists. It supports:
- **CSV** input with configurable delimiter
- **PKL** input

The input graph is converted to an `STensor`, then processed with `st.Fraudar`.

## Workflow

### Step 1: Understand Requirements

Identify:
- File location
- File format (CSV / PKL)
- Separator
- Header
- Has value
- k (number of suspicious blocks)
- maxsize (optional constraint)

### Step 2: Run Analysis

```bash
python my_skills/Fraudar/scripts/analyze.py \
  --files plain_graph_small.csv \
  --sep "," \
  --header none \
  --hasvalue false \
  --k 2 \
  --maxsize 100 100