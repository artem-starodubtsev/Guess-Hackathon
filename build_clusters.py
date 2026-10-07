import time

import cv2
import numpy as np
from umap import UMAP

from common import ROOT, ARTIFACTS, read_json, write_json


def main():
    started = time.perf_counter()
    catalog = read_json(ARTIFACTS / 'catalog.json')
    items = catalog['items']
    n = len(items)
    vectors = np.asarray([x['descriptor'] for x in items], np.float32)
    palette = np.zeros((n, n), np.float32)
    signatures = [np.asarray(x['palette'], np.float32) for x in items]
    for i in range(n):
        for j in range(i):
            palette[i, j] = palette[j, i] = cv2.EMD(signatures[i], signatures[j], cv2.DIST_L2)[0]
    matrices = [palette]
    for block in (vectors[:, 16:80], vectors[:, 80:]):
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        matrices.append(np.clip(1 - block @ block.T, 0, 2))
    matrices = [m / scale for m, scale in zip(matrices, catalog['config']['scales'])]
    result = {'items': [{k: x[k] for k in ('id','name','style','color','category')} for x in items], 'maps': {}}
    for bits in range(1, 8):
        indices = [j for j in range(3) if bits & (1 << j)]
        distance = np.mean([matrices[j] for j in indices], axis=0)
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        xy = UMAP(metric='precomputed', n_neighbors=25, min_dist=0.12, random_state=42, n_jobs=1).fit_transform(distance)
        neighbors = []
        for i, row in enumerate(distance):
            seen = {(items[i]['style'], items[i]['color'])}
            chosen = []
            for j in np.argsort(row, kind='stable'):
                key = (items[j]['style'], items[j]['color'])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append([int(j), round(float(row[j]), 6)])
                if len(chosen) == 8:
                    break
            neighbors.append(chosen)
        result['maps'][str(bits)] = {'xy': xy.round(5).tolist(), 'neighbors': neighbors}
        print(f'Map {bits}/7 ready', flush=True)
    result['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    write_json(ROOT / 'web' / 'clusters-data.json', result)
    print(f"Done in {result['elapsed_seconds']}s", flush=True)


if __name__ == '__main__':
    main()
