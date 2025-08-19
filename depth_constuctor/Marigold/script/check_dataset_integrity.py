#!/usr/bin/env python
import argparse, json, csv, sys, itertools, pathlib, traceback
from PIL import Image
import numpy as np

def png_jpg_stats(path):
    try:
        im = Image.open(path)
        arr = np.asarray(im)
    except Exception as e:
        return {'error': str(e)}
    stats = base_stats(arr)
    stats['mode'] = im.mode
    return stats

def npy_npz_stats(path):
    try:
        if path.suffix == '.npy':
            arr = np.load(path, allow_pickle=False)
        else:                                 # npz
            with np.load(path, allow_pickle=False) as zf:
                # handle both single‑array and dict‑like archives
                keys = list(zf.keys())
                if len(keys) == 1:
                    arr = zf[keys[0]]
                else:
                    return {'multiple_arrays': keys}
    except Exception as e:
        return {'error': str(e)}
    return base_stats(arr)

def base_stats(arr, print_every=None, index=None):
    finite = np.isfinite(arr)
    obj = {
        'shape'   : arr.shape,
        'dtype'   : str(arr.dtype),
        'has_nan' : (np.isnan(arr).any()).item(),
        'has_inf' : (np.isinf(arr).any()).item(),
        'is_const': (arr[finite].size and
                     np.all(arr[finite] == arr[finite].flat[0])).item(),
        'min'     : float(np.nanmin(arr)),
        'max'     : float(np.nanmax(arr)),
        'mean'    : float(np.nanmean(arr)),
    }

    if print_every is not None and index is not None and index % print_every == 0:
        print(f"[{index}] stats = {obj}")

    return obj



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True,
                    help='dataset root folders to scan')
    ap.add_argument('--out', default='integrity_report.csv')
    args = ap.parse_args()

    records = []
    for root in args.roots:
        root = pathlib.Path(root)
        if not root.exists():
            print(f'[WARN] {root} does not exist', file=sys.stderr)
            continue

        for i, path in enumerate(root.rglob('*')):
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                rec = png_jpg_stats(path)
            elif path.suffix.lower() in ('.npy', '.npz'):
                rec = npy_npz_stats(path)
            else:
                continue

            rec['file'] = str(path)
            records.append(rec)

            # print the full stats every 5 files
            if i % 5 == 0:
                print(f"\n[{i}] File: {path}")
                for k, v in rec.items():
                    print(f"  {k:<10s}: {v}")


    # write CSV
    keys = sorted(set(itertools.chain.from_iterable(r.keys() for r in records)))
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)

    print(f'Wrote {len(records)} lines to {args.out}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass







