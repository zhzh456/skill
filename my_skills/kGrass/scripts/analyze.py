"""kGrass graph summarization skill."""

import argparse
import json
import sys

import numpy as np
import spartan as st


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def load_graph(file_path, sep="\t", hasvalue=False):
    data = st.loadTensor(
        path=file_path,
        header=None,
        col_idx=[0, 1],
        col_types=[int, int],
        sep=sep,
    )
    mapper = st.DenseIntMapper()
    tensor = data.toSTensor(hasvalue=hasvalue, mappers={0: mapper, 1: mapper})
    n = mapper._idx
    tensor.shape = (n, n)
    graph = st.Graph(tensor)
    return graph, n


def run_kgrass(graph, K=1000, strategy="sample_pairs"):
    model = st.kGrass(graph)
    return model.run(K=K, strategy=strategy)


def summarize_adj(adj):
    if adj is None:
        return None
    if hasattr(adj, "shape"):
        nnz = int(adj.nnz) if hasattr(adj, "nnz") else None
        return {"shape": list(adj.shape), "nnz": nnz}
    return str(adj)


def main():
    parser = argparse.ArgumentParser(description="Run kGrass graph summarization")
    parser.add_argument("--file", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--K", type=int, default=1000, help="Summary graph size (must be < num nodes)")
    parser.add_argument(
        "--strategy",
        default="sample_pairs",
        choices=["greedy", "linear_check", "sample_pairs"],
    )
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    output = {"algorithm": "kGrass", "file": args.file}

    try:
        hasvalue = parse_bool(args.hasvalue)
        graph, n = load_graph(args.file, sep=args.sep, hasvalue=hasvalue)
        if args.K >= n:
            raise ValueError(f"K ({args.K}) must be less than graph size ({n})")
        adj = run_kgrass(graph, K=args.K, strategy=args.strategy)
        output["graph_size"] = n
        output["result"] = summarize_adj(adj)
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
