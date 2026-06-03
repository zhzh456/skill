"""CubeFlow fraud detection skill."""

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


def load_tensor(path):
    return st.loadTensor(path=path, header=None)


def run_cubeflow(amt_tensor, cmt_tensor, alpha=0.8, k=1, dim=3, outpath=""):
    amt_stensor = amt_tensor.toSTensor(hasvalue=True)
    cmt_stensor = cmt_tensor.toSTensor(hasvalue=True)

    model = st.CubeFlow(
        [amt_stensor, cmt_stensor],
        alpha=alpha,
        k=k,
        dim=dim,
        outpath=outpath
    )

    res = model.run(del_type=1, maxsize=-1)
    return res


def get_groundtruth(path):
    gt = np.load(path, allow_pickle=True)
    return gt.tolist()


def find_top_k_res(real_res, k):
    top_k_a, top_k_m, top_k_c = set(), set(), set()

    for i in range(k):
        top_k_a |= set(real_res[i][0][0])
        top_k_m |= set(real_res[i][0][1])
        top_k_c |= set(real_res[i][0][2])

    return [top_k_a, top_k_m, top_k_c]


def cal_f1(detectSet, trueSet, amt_tensor, cmt_tensor):
    fs1 = amt_tensor.data
    fs2 = cmt_tensor.data

    set_a = detectSet[0] & trueSet[0]
    set_m = detectSet[1] & trueSet[1]
    set_c = detectSet[2] & trueSet[2]

    tp1 = fs1[(fs1[0].isin(set_a)) & (fs1[1].isin(set_m))]
    tp2 = fs2[(fs2[1].isin(set_m)) & (fs2[0].isin(set_c))]

    fptp1 = fs1[(fs1[0].isin(detectSet[0])) & (fs1[1].isin(detectSet[1]))]
    fptp2 = fs2[(fs2[1].isin(detectSet[1])) & (fs2[0].isin(detectSet[2]))]

    fntp1 = fs1[(fs1[0].isin(trueSet[0])) & (fs1[1].isin(trueSet[1]))]
    fntp2 = fs2[(fs2[1].isin(trueSet[1])) & (fs2[0].isin(trueSet[2]))]

    tpm = tp1[3].sum() + tp2[3].sum()
    tpfpm = fptp1[3].sum() + fptp2[3].sum()
    tpfnm = fntp1[3].sum() + fntp2[3].sum()

    precision = tpm / tpfpm if tpfpm > 0 else 0
    recall = tpm / tpfnm if tpfnm > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"f1": f1, "precision": precision, "recall": recall}


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
    parser = argparse.ArgumentParser(description="Run CubeFlow fraud detection")

    parser.add_argument("--amt-file", required=True)
    parser.add_argument("--cmt-file", required=True)
    parser.add_argument("--gt-file", default=None)

    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dim", type=int, default=3)

    parser.add_argument("--output-file", default=None)

    args = parser.parse_args()

    try:
        amt_tensor = load_tensor(args.amt_file)
        cmt_tensor = load_tensor(args.cmt_file)

        res = run_cubeflow(
            amt_tensor,
            cmt_tensor,
            alpha=args.alpha,
            k=args.k,
            dim=args.dim
        )

        output = {
            "algorithm": "CubeFlow",
            "result": make_json_safe(res)
        }

        if args.gt_file:
            gt = get_groundtruth(args.gt_file)
            metrics = []

            for i in range(1, args.k + 1):
                top_k = find_top_k_res(res, i)
                score = cal_f1(top_k, gt, amt_tensor, cmt_tensor)
                metrics.append({
                    "top_k": i,
                    **score
                })

            output["evaluation"] = metrics

    except Exception as e:
        output = {
            "algorithm": "CubeFlow",
            "error": str(e)
        }

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()