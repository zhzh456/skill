"""FlowScope money-laundering flow detection skill."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import spartan as st


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def load_bipartite_tensor(path, four_dim=False):
    """Load a 3-column (src, dst, value) tensor for FlowScope."""
    path = Path(path)
    load_path = path

    if four_dim:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 4:
            raise ValueError(f"--four-dim expects >=4 columns, got {df.shape[1]} in {path}")
        projected = df.iloc[:, [0, 1, df.shape[1] - 1]]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        projected.to_csv(tmp.name, header=False, index=False)
        tmp.close()
        load_path = tmp.name
    elif path.suffix == ".csv" and not path.name.endswith(".flowscope_tmp.csv"):
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 4:
            projected = df.iloc[:, [0, 1, 3]]
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            projected.to_csv(tmp.name, header=False, index=False)
            tmp.close()
            load_path = tmp.name

    tensor_data = st.loadTensor(path=str(load_path), header=None)
    return tensor_data.toSTensor(hasvalue=True)


def build_graph_list(fs1_path, fs2_path, four_dim=False):
    fs1 = load_bipartite_tensor(fs1_path, four_dim=four_dim)
    fs2 = load_bipartite_tensor(fs2_path, four_dim=four_dim)

    maxshape = max(fs1.shape[1], fs2.shape[0])
    fs1.shape = (fs1.shape[0], maxshape)
    fs2.shape = (maxshape, fs2.shape[1])

    graph_1 = st.Graph(fs1, bipartite=True, weighted=True, modet=None)
    graph_2 = st.Graph(fs2, bipartite=True, weighted=True, modet=None)
    return [graph_1, graph_2]


def run_flowscope(graph_list, k=3, alpha=4, maxsize=None):
    fs = st.FlowScope(graph_list)
    if maxsize is None:
        return fs.run(k=k, alpha=alpha)
    return fs.run(k=k, alpha=alpha, maxsize=tuple(maxsize))


def make_json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run FlowScope on coupled bipartite graphs")
    parser.add_argument("--fs1-file", required=True)
    parser.add_argument("--fs2-file", required=True)
    parser.add_argument("--four-dim", action="store_true", help="Project 4-column CFD-style CSV to 3 columns (0,1,last)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=4)
    parser.add_argument("--maxsize", nargs=3, type=int, default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    output = {
        "algorithm": "FlowScope",
        "fs1_file": args.fs1_file,
        "fs2_file": args.fs2_file,
    }

    try:
        graphs = build_graph_list(args.fs1_file, args.fs2_file, four_dim=args.four_dim)
        result = run_flowscope(graphs, k=args.k, alpha=args.alpha, maxsize=args.maxsize)
        output["result"] = make_json_safe(result)
    except Exception as exc:
        output["error"] = str(exc)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
