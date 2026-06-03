"""SpecGreedy dense subgraph detection skill."""

import argparse
import json
import pickle
import sys

import spartan as st


def parse_header(value):
    if value == "none":
        return None
    if value == "infer":
        return "infer"
    raise ValueError("header must be 'none' or 'infer'")


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def load_stensor(file_path, sep=" ", header="none", hasvalue=False, col_types=None):
    if file_path.lower().endswith(".pkl"):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if hasattr(data, "toSTensor"):
            return data.toSTensor(hasvalue=hasvalue)
        return data

    parsed_header = parse_header(header)
    kwargs = {"path": file_path, "header": parsed_header, "sep": sep}
    if col_types:
        kwargs["col_types"] = col_types
    tensor = st.loadTensor(**kwargs)
    return tensor.toSTensor(hasvalue=hasvalue)


def run_specgreedy(stensor, bipartite=True, T=5):
    sg = st.Specgreedy(stensor)
    return sg.run(bipartite=bipartite, T=T)


def make_json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run SpecGreedy dense subgraph detection")
    parser.add_argument("--file", required=True)
    parser.add_argument("--sep", default=" ")
    parser.add_argument("--header", default="none")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--bipartite", default="true")
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    output = {"algorithm": "SpecGreedy", "file": args.file}

    try:
        hasvalue = parse_bool(args.hasvalue)
        bipartite = parse_bool(args.bipartite)
        stensor = load_stensor(args.file, sep=args.sep, header=args.header, hasvalue=hasvalue, col_types=[int, int])
        result = run_specgreedy(stensor, bipartite=bipartite, T=args.T)
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
