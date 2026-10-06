"""Build ROI features from an authorized expert-labelled split manifest."""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from policy import CLASSES
from roi_features import feature_one


def prepare(manifest, image_root, output):
    groups = {s: [] for s in ('train', 'validation', 'test')}
    hashes = set()
    base = image_root.resolve()
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get('source') != 'expert':
            raise ValueError('Primary classifier admits expert-labelled records only')
        split, label = row['split'], row['label']
        if split not in groups or label not in CLASSES:
            raise ValueError('Invalid split or label')
        rel = Path(row['path'])
        path = (base / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(base):
            raise ValueError('Image path must be relative and within image root')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in hashes:
            raise ValueError('Exact duplicate image found; adjudicate/deduplicate before training')
        hashes.add(sha)
        groups[split].append((feature_one(path), CLASSES.index(label)))
    payload = {}
    for split, key in [('train','tr'),('validation','v'),('test','t')]:
        if not groups[split]:
            raise ValueError(f'Empty partition: {split}')
        payload['X'+key] = np.stack([x for x,y in groups[split]])
        payload['y'+key] = np.array([y for x,y in groups[split]], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return {s: len(group) for s, group in groups.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try: print(prepare(a.manifest, a.image_root, a.output))
    except (ValueError, KeyError, OSError) as e: p.error(str(e))
