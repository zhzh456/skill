"""EagleMine vision-guided micro-cluster detection skill."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import spartan as st


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("boolean must be true/false")


def check_eaglemine_import():
    try:
        import pomegranate  # noqa: F401
        from pomegranate import GeneralMixtureModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "EagleMine requires pomegranate 0.14.x (GeneralMixtureModel). "
            "Install with: pip install 'pomegranate==0.14.9' "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc


def ensure_out_dir(out_dir):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_eaglemine_zip(zip_path, dest_dir):
    zpath = Path(zip_path)
    if not zpath.is_file():
        return
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(dest_dir)


def load_histogram_inputs(histogram, node2hcel, hcel2avgfeat):
    histogram_dict = st.loadFile2Dict(histogram, 2, int, int)
    node2hcel_dict = st.loadFile2Dict(node2hcel, 1, int, int)
    hcel2avgfeat_dict = st.loadFile2Dict(hcel2avgfeat, 2, int, float)
    return histogram_dict, node2hcel_dict, hcel2avgfeat_dict


def build_histogram_from_graph(
    eaglemine,
    graph_file,
    sep=",",
    hasvalue=False,
    feature_type="outdegree2hubness",
    n_bins=80,
    base=10,
    mode=2,
):
    tensor_data = st.loadTensor(path=graph_file, header=None, sep=sep)
    stensor = tensor_data.toSTensor(hasvalue=hasvalue)
    graph = st.Graph(stensor, bipartite=True, weighted=True, modet=2)
    degreeidx, feature = eaglemine.graph2feature(graph, feature_type)
    return eaglemine.feature2histogram(
        feature, degreeidx, N_bins=n_bins, base=base, mode=mode, verbose=False
    )


def run_eaglemine(args):
    check_eaglemine_import()

    eaglemine = st.EagleMine(
        voctype=args.voctype,
        mode=args.mode,
        mix_comps=args.mix_comps,
    )

    out_dir = ensure_out_dir(args.out_dir)

    if args.eaglemine_zip:
        extract_eaglemine_zip(args.eaglemine_zip, Path(args.histogram).parent)

    if args.input_mode == "histogram":
        histogram, node2hcel, hcel2avgfeat = load_histogram_inputs(
            args.histogram, args.node2hcel, args.hcel2avgfeat
        )
    elif args.input_mode == "graph":
        histogram, node2hcel, hcel2avgfeat = build_histogram_from_graph(
            eaglemine,
            args.graph_file,
            sep=args.sep,
            hasvalue=parse_bool(args.hasvalue),
            feature_type=args.feature_type,
            n_bins=args.n_bins,
            base=args.base,
            mode=args.mode,
        )
        if args.save_histogram:
            eaglemine.save_histogram(
                str(out_dir / "histogram.out"),
                str(out_dir / "node2hcel.out"),
                str(out_dir / "hcel2avgfeat.out"),
                comments="#",
                delimiter=",",
            )
    else:
        raise ValueError("input_mode must be 'histogram' or 'graph'")

    eaglemine.set_histdata(histogram, node2hcel, hcel2avgfeat, weighted_ftidx=0)
    eaglemine.run(
        outs=str(out_dir) + "/",
        waterlevel_step=args.waterlevel_step,
        prune_alpha=args.prune_alpha,
        min_pts=args.min_pts,
        strictness=args.strictness,
        verbose=args.verbose,
    )

    if args.save_results:
        eaglemine.save(
            str(out_dir / "eaglemine.out"),
            str(out_dir / "waterleveltree.out"),
            str(out_dir / "node2label.out"),
            str(out_dir / "hcel2label.out"),
            comments="#",
            delimiter=",",
        )

    return {"out_dir": str(out_dir), "status": "completed"}


def main():
    parser = argparse.ArgumentParser(description="Run EagleMine anomaly detection")
    parser.add_argument("--input-mode", choices=["histogram", "graph"], required=True)

    parser.add_argument("--histogram", default="./inputData/histogram.out")
    parser.add_argument("--node2hcel", default="./inputData/node2hcel.out")
    parser.add_argument("--hcel2avgfeat", default="./inputData/hcel2avgfeat.out")
    parser.add_argument("--eaglemine-zip", default="./inputData/eaglemine_data.zip")

    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--hasvalue", default="false")
    parser.add_argument("--feature-type", default="outdegree2hubness")
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--save-histogram", action="store_true")

    parser.add_argument("--voctype", default="dtmnorm")
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--mix-comps", type=int, default=2)

    parser.add_argument("--out-dir", default="./output")
    parser.add_argument("--waterlevel-step", type=float, default=0.2)
    parser.add_argument("--prune-alpha", type=float, default=0.80)
    parser.add_argument("--min-pts", type=int, default=20)
    parser.add_argument("--strictness", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-file", default=None)

    args = parser.parse_args()

    if args.input_mode == "graph" and not args.graph_file:
        print("--graph-file is required when --input-mode graph", file=sys.stderr)
        sys.exit(1)

    output = {"algorithm": "EagleMine", "input_mode": args.input_mode}

    try:
        output["result"] = run_eaglemine(args)
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
