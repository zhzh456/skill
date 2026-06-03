"""DPGS graph processing skill."""

import argparse
import json
import sys

import spartan as st


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def load_tensor(file_path, sep="\t"):
    # load edge list tensor
    return st.loadTensor(
        path=file_path,
        header=None,
        col_idx=[0, 1],
        col_types=[int, int],
        sep=sep
    )


def run_dpgs(tensor, hasvalue=False, T=5):
    mapper = st.DenseIntMapper()

    stensor = tensor.toSTensor(
        hasvalue=hasvalue,
        mappers={0: mapper, 1: mapper}
    )

    N = mapper._idx
    stensor.shape = (N, N)

    graph = st.Graph(stensor)
    model = st.DPGS(graph)

    return model.run(T=T)


def make_json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run DPGS on graph data")

    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--output-file", default=None)

    args = parser.parse_args()

    hasvalue = parse_bool(args.hasvalue)

    results = []

    for file_path in args.files:
        try:
            tensor = load_tensor(file_path, sep=args.sep)
            adj_s = run_dpgs(tensor, hasvalue=hasvalue, T=args.T)

            results.append({
                "file": file_path,
                "algorithm": "DPGS",
                "result": make_json_safe(adj_s)
            })

        except Exception as e:
            results.append({
                "file": file_path,
                "algorithm": "DPGS",
                "error": str(e)
            })

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()