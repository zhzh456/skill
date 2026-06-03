"""MonLAD money laundering agent detection skill."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import spartan as st
from spartan.tensor import TensorStream

# vendored MonLAD (not in spartan2 0.1.3.post4)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from monlad.MonLAD import MonLAD  # noqa: E402


def apply_tensor_stream_patch():
    _orig = TensorStream.fetch_slide_window

    def fetch_slide_window(self, window=10, stride=5, ts_colidx=0, decode=False, **kwargs):
        return _orig(self, window, stride, ts_colidx)

    TensorStream.fetch_slide_window = fetch_slide_window


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def parse_col_types(spec):
    mapping = {"int": int, "str": str, "float": float}
    return [mapping[x.lower()] for x in spec.split(",")]


def open_stream(path):
    return open(path, "rb")


def run_monlad(
    file_path,
    sep=",",
    col_idx=None,
    col_types=None,
    hasvalue=True,
    delta_up=10000,
    delta_down=10000,
    epsilon=10000,
    window=1,
    stride=1,
    ts_idx=1,
    has_edge=False,
    source_type="VYDAJ",
    des_type="PRIJEM",
    run_anomaly=False,
    detect_part=None,
    alpha=0.5,
    k=1.0,
    p=0.8,
    outpath=None,
):
    apply_tensor_stream_patch()

    if col_idx is None:
        col_idx = [0, 1, 2, 3]
    if col_types is None:
        col_types = [int, int, str, float] if not has_edge else [int, int, int, float]

    f = open_stream(file_path)
    try:
        tensor_stream = TensorStream(
            f,
            col_idx=col_idx,
            col_types=col_types,
            sep=sep,
            mappers={},
            hasvalue=hasvalue,
        )
        params = {
            "deltaUp": delta_up,
            "deltaDown": delta_down,
            "epsilon": epsilon,
            "window": window,
            "stride": stride,
            "ts_idx": ts_idx,
            "has_edge": has_edge,
            "source_type": source_type,
            "des_type": des_type,
        }
        monlad = MonLAD(tensor_stream, **params)
        count_df = monlad.run()

        output = {
            "num_accounts": int(len(count_df)),
            "count_stats": {
                "count_mean": float(count_df["count"].mean()) if len(count_df) else 0,
                "countIn_mean": float(count_df["countIn"].mean()) if len(count_df) else 0,
            },
            "preview": count_df.head(10).to_dict(orient="records"),
        }

        if run_anomaly:
            parts = detect_part or [1, 2, 3, 4]
            anomalous = monlad.anomaly_detection(
                detect_part=parts, alpha=alpha, k=k, p=p, outpath=outpath
            )
            output["anomalous_accounts"] = make_json_safe(anomalous)

        return output
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
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run MonLAD on transaction stream data")
    parser.add_argument("--file", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--col-idx", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--col-types",
        default="int,int,str,float",
        help="Comma-separated types: int,str,float",
    )
    parser.add_argument("--hasvalue", default="true")
    parser.add_argument("--delta-up", type=float, default=10000)
    parser.add_argument("--delta-down", type=float, default=10000)
    parser.add_argument("--epsilon", type=float, default=10000)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ts-idx", type=int, default=1)
    parser.add_argument("--has-edge", default="false")
    parser.add_argument("--source-type", default="VYDAJ")
    parser.add_argument("--des-type", default="PRIJEM")
    parser.add_argument("--run-anomaly", action="store_true")
    parser.add_argument("--detect-part", nargs="+", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    output = {"algorithm": "MonLAD", "file": args.file}

    try:
        output["result"] = run_monlad(
            args.file,
            sep=args.sep,
            col_idx=args.col_idx,
            col_types=parse_col_types(args.col_types),
            hasvalue=parse_bool(args.hasvalue),
            delta_up=args.delta_up,
            delta_down=args.delta_down,
            epsilon=args.epsilon,
            window=args.window,
            stride=args.stride,
            ts_idx=args.ts_idx,
            has_edge=parse_bool(args.has_edge),
            source_type=args.source_type,
            des_type=args.des_type,
            run_anomaly=args.run_anomaly,
            detect_part=args.detect_part,
            alpha=args.alpha,
            k=args.k,
            p=args.p,
            outpath=args.out_dir,
        )
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
