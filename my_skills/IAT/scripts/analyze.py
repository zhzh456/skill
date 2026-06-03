"""IAT (Inter-Arrival Time) micro-cluster analysis skill."""

import argparse
import json
import os
import sys

import spartan as st


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def load_aggts(
    file_path,
    sep="\x01",
    hasvalue=False,
    time_col=0,
    group_col=None,
    time_format="%Y-%m-%d %H:%M:%S",
    timebin=1,
):
    if group_col is None:
        group_col = [1]

    tensor_data = st.loadTensor(path=file_path, header=None, sep=sep)
    if time_format in {"int", "none", "raw"}:
        coords, _ = tensor_data.do_map(hasvalue=hasvalue, mappers={})
    else:
        mappers = {time_col: st.TimeMapper(timeformat=time_format, timebin=timebin, mints=0)}
        coords, _ = tensor_data.do_map(hasvalue=hasvalue, mappers=mappers)
    return tensor_data.to_aggts(coords, time_col=time_col, group_col=group_col)


def run_iat(
    file_path,
    sep="\x01",
    hasvalue=False,
    time_col=0,
    group_col=None,
    time_format="%Y-%m-%d %H:%M:%S",
    timebin=1,
    save_aggiat=None,
    load_aggiat=None,
    top_k=10,
):
    instance = st.IAT()

    if load_aggiat:
        instance.load_aggiat(load_aggiat)
    else:
        aggts = load_aggts(
            file_path,
            sep=sep,
            hasvalue=hasvalue,
            time_col=time_col,
            group_col=group_col,
            time_format=time_format,
            timebin=timebin,
        )
        instance.calaggiat(aggts)
        if save_aggiat:
            os.makedirs(os.path.dirname(save_aggiat) or ".", exist_ok=True)
            instance.save_aggiat(save_aggiat)

    xs, ys = instance.getiatpairs()
    instance.get_user_iatpair_dict()
    instance.get_iatpair_user_dict()
    instance.caliatcount()
    instance.caliatpaircount()

    top_users = []

    return {
        "num_users": len(instance.aggiat),
        "num_iat_pairs": len(xs),
        "top_users": make_json_safe(top_users),
        "iat_count_size": len(instance.iatcount),
        "iat_pair_count_size": len(instance.iatpaircount),
    }


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
    parser = argparse.ArgumentParser(description="Run IAT inter-arrival time analysis")
    parser.add_argument("--file", default=None, help="Event log / edge list with timestamps")
    parser.add_argument("--load-aggiat", default=None, help="Load precomputed aggiat .dictlist.gz")
    parser.add_argument("--save-aggiat", default=None, help="Save computed aggiat to file")
    parser.add_argument("--sep", default="\x01")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--time-col", type=int, default=0)
    parser.add_argument("--group-col", nargs="+", type=int, default=[1])
    parser.add_argument("--time-format", default="%Y-%m-%d %H:%M:%S")
    parser.add_argument("--timebin", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    if not args.file and not args.load_aggiat:
        print("Either --file or --load-aggiat is required", file=sys.stderr)
        sys.exit(1)

    output = {"algorithm": "IAT", "file": args.file or args.load_aggiat}

    try:
        output["result"] = run_iat(
            args.file,
            sep=args.sep,
            hasvalue=parse_bool(args.hasvalue),
            time_col=args.time_col,
            group_col=args.group_col,
            time_format=args.time_format,
            timebin=args.timebin,
            save_aggiat=args.save_aggiat,
            load_aggiat=args.load_aggiat,
            top_k=args.top_k,
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
