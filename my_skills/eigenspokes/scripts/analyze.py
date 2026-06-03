"""Eigenspokes graph processing skill."""

import argparse
import json
import pickle
import sys

import spartan as st


def parse_header(value):
    if value == 'none':
        return None
    if value == 'infer':
        return 'infer'
    raise ValueError("header must be 'none' or 'infer'")


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes'}:
        return True
    if value in {'false', '0', 'no'}:
        return False
    raise ValueError("boolean value must be one of: true, false, 1, 0, yes, no")


def load_input(file_path, sep=',', header='none'):
    if file_path.lower().endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    parsed_header = parse_header(header)
    return st.loadTensor(path=file_path, header=parsed_header, sep=sep)


def run_eigenspokes(data, hasvalue=False):
    if hasattr(data, 'toSTensor'):
        stensor = data.toSTensor(hasvalue=hasvalue)
    else:
        stensor = data

    es = st.Eigenspokes(stensor)
    return es.run()


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
    parser = argparse.ArgumentParser(description='Run Eigenspokes on graph data')
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--sep', default=',')
    parser.add_argument('--header', default='none')
    parser.add_argument('--hasvalue', default='false')
    parser.add_argument('--output-file', default=None)
    args = parser.parse_args()

    try:
        hasvalue = parse_bool(args.hasvalue)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    all_results = []

    for file_path in args.files:
        try:
            data = load_input(file_path, sep=args.sep, header=args.header)
            result = run_eigenspokes(data, hasvalue=hasvalue)

            all_results.append({
                'file': file_path,
                'algorithm': 'Eigenspokes',
                'result': make_json_safe(result)
            })
        except Exception as e:
            all_results.append({
                'file': file_path,
                'algorithm': 'Eigenspokes',
                'error': str(e)
            })

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()