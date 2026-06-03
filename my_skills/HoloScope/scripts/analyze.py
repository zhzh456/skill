"""HoloScope graph fraud detection skill."""

import argparse
import json
import sys

import numpy as np

import spartan as st


def apply_numpy_ragged_patch():
    """HoloScope in spartan fails on NumPy>=1.24 when building ragged CU arrays."""
    _orig = np.array

    def _patched(obj, *args, **kwargs):
        if "dtype" not in kwargs and isinstance(obj, (list, tuple)) and len(obj) > 0:
            try:
                return _orig(obj, *args, **kwargs)
            except ValueError as exc:
                if "inhomogeneous" in str(exc) or "sequence" in str(exc):
                    return _orig(obj, dtype=object, *args, **kwargs)
        return _orig(obj, *args, **kwargs)

    np.array = _patched


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
    raise ValueError("boolean value must be one of: true, false, 1, 0, yes, no")


def load_graph(file_path, sep=",", header="none", hasvalue=False, time_col=None, time_format="%Y-%m-%d"):
    parsed_header = parse_header(header)
    tensor_data = st.loadTensor(path=file_path, header=parsed_header, sep=sep)

    mappers = {}
    if time_col is not None:
        mappers[int(time_col)] = st.TimeMapper(timeformat=time_format)

    stensor = tensor_data.toSTensor(hasvalue=hasvalue, mappers=mappers)
    modet = 2 if hasvalue or mappers else 2
    graph = st.Graph(stensor, bipartite=True, weighted=True, modet=modet)
    return graph


def run_holoscope(graph, level=0, k=1):
    hs = st.HoloScope(graph)
    return hs.run(level=level, k=k)


def make_json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run HoloScope on graph data")
    parser.add_argument("--file", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--header", default="none")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--time-col", type=int, default=None)
    parser.add_argument("--time-format", default="%Y-%m-%d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    apply_numpy_ragged_patch()

    try:
        hasvalue = parse_bool(args.hasvalue)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    output = {"algorithm": "HoloScope", "file": args.file}

    try:
        graph = load_graph(
            args.file,
            sep=args.sep,
            header=args.header,
            hasvalue=hasvalue,
            time_col=args.time_col,
            time_format=args.time_format,
        )
        result = run_holoscope(graph, level=args.level, k=args.k)
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
