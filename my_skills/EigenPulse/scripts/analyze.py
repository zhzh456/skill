"""EigenPulse streaming graph surge detection skill."""

import argparse
import gzip
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
import spartan as st
from spartan.tensor import STensor, TensorData
from spartan.util.basicutil import StringMapper


def apply_spartan_compat_patches():
    """Fix StringMapper.map and TensorData.toSTensor for EigenPulse tuple columns."""

    def fixed_string_map(self, attrs):
        out = []
        for s in attrs:
            if s not in self.strdict:
                self.strdict[s] = len(self.strdict)
                self.strids.append(s)
            out.append(self.strdict[s])
        return out

    StringMapper.map = fixed_string_map

    def fixed_to_stensor(self, hasvalue=True, mappers=None):
        mappers = mappers or {}
        if hasvalue:
            value = self.data.iloc[:, -1]
            attr = self.data.iloc[:, :-1].copy()
        else:
            value = pd.Series([1] * len(self.data))
            attr = self.data.copy()

        for i in attr.columns:
            if i in mappers:
                colind = mappers[i].map(self.data.iloc[:, i])
                attr[i] = pd.Series(colind, dtype=np.int64)
            else:
                attr[i] = attr[i].astype(np.int64)

        return STensor((attr.to_numpy().T, value.to_numpy()))

    TensorData.toSTensor = fixed_to_stensor


def open_stream(path):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def run_eigenpulse(
    file_path,
    col_idx,
    col_types,
    sep=",",
    hasvalue=True,
    window=20,
    stride=10,
    l=20,
    b=10,
    item_idx=1,
    ts_idx=2,
):
    apply_spartan_compat_patches()

    f = open_stream(file_path)
    try:
        tensor_stream = st.TensorStream(
            f,
            col_idx=col_idx,
            col_types=col_types,
            sep=sep,
            mappers={},
            hasvalue=hasvalue,
        )
        eigenpulse = st.EigenPulse(
            tensor_stream,
            window=window,
            stride=stride,
            l=l,
            b=b,
            item_idx=item_idx,
            ts_idx=ts_idx,
        )
        return eigenpulse.run()
    finally:
        f.close()


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
    parser = argparse.ArgumentParser(description="Run EigenPulse on streaming tensor data")
    parser.add_argument("--file", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--col-idx", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--hasvalue", default="true")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--l", type=int, default=20)
    parser.add_argument("--b", type=int, default=10)
    parser.add_argument("--item-idx", type=int, default=1)
    parser.add_argument("--ts-idx", type=int, default=2)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    hasvalue = str(args.hasvalue).strip().lower() in {"true", "1", "yes"}
    col_types = [int] * len(args.col_idx)

    output = {"algorithm": "EigenPulse", "file": args.file}

    try:
        res, densities = run_eigenpulse(
            args.file,
            col_idx=args.col_idx,
            col_types=col_types,
            sep=args.sep,
            hasvalue=hasvalue,
            window=args.window,
            stride=args.stride,
            l=args.l,
            b=args.b,
            item_idx=args.item_idx,
            ts_idx=args.ts_idx,
        )
        output["result"] = make_json_safe(res)
        output["densities"] = make_json_safe(densities)
        output["num_windows"] = len(densities) if densities else 0
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
