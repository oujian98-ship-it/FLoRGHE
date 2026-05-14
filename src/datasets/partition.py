from __future__ import annotations

import numpy as np


def dirichlet_partition(labels, num_clients, alpha, seed=0, min_size=1):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
            for cid, part in enumerate(np.split(idx_c, cuts)):
                client_indices[cid].extend(part.tolist())
        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            break

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def client_label_stats(labels, client_indices):
    labels = np.asarray(labels)
    stats = []
    for cid, indices in enumerate(client_indices):
        unique, counts = np.unique(labels[indices], return_counts=True) if indices else ([], [])
        stats.append(
            {
                "client_id": cid,
                "num_samples": len(indices),
                "label_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            }
        )
    return stats
