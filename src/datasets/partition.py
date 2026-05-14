from __future__ import annotations

import numpy as np


def _num_classes_in_client(labels, indices):
    if len(indices) == 0:
        return 0
    return len(np.unique(labels[indices]))


def _is_valid_partition(
    labels,
    client_indices,
    min_size=1,
    min_classes_per_client=1,
    max_size_ratio=None,
):
    sizes = [len(x) for x in client_indices]

    if min(sizes) < min_size:
        return False

    if min_classes_per_client is not None and min_classes_per_client > 1:
        for indices in client_indices:
            if _num_classes_in_client(labels, np.asarray(indices)) < min_classes_per_client:
                return False

    if max_size_ratio is not None:
        min_s = max(1, min(sizes))
        max_s = max(sizes)
        if max_s / min_s > max_size_ratio:
            return False

    return True


def iid_partition(labels, num_clients, seed=0, min_size=1):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    parts = np.array_split(indices, num_clients)
    client_indices = [p.tolist() for p in parts]

    sizes = [len(x) for x in client_indices]
    if min(sizes) < min_size:
        raise ValueError(
            f"IID partition failed: min client size={min(sizes)} < min_size={min_size}. "
            "Reduce num_clients or min_size."
        )

    return client_indices


def dirichlet_partition(
    labels,
    num_clients,
    alpha,
    seed=0,
    min_size=1,
    min_classes_per_client=1,
    max_size_ratio=None,
    max_tries=5000,
):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    if min_classes_per_client > len(classes):
        raise ValueError(
            f"min_classes_per_client={min_classes_per_client} is larger than "
            f"number of classes={len(classes)}"
        )

    last_sizes = None
    last_label_stats = None

    for _ in range(max_tries):
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)

            for cid, part in enumerate(np.split(idx_c, cuts)):
                client_indices[cid].extend(part.tolist())

        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])

        if _is_valid_partition(
            labels=labels,
            client_indices=client_indices,
            min_size=min_size,
            min_classes_per_client=min_classes_per_client,
            max_size_ratio=max_size_ratio,
        ):
            return client_indices

        last_sizes = [len(x) for x in client_indices]
        last_label_stats = client_label_stats(labels, client_indices)

    raise RuntimeError(
        "Failed to generate a valid Dirichlet partition. "
        f"alpha={alpha}, num_clients={num_clients}, min_size={min_size}, "
        f"min_classes_per_client={min_classes_per_client}, "
        f"max_size_ratio={max_size_ratio}, max_tries={max_tries}. "
        f"Last sizes={last_sizes}. "
        "Try increasing alpha, reducing num_clients, reducing min_size, "
        "or setting min_classes_per_client=1. "
        f"Last label stats={last_label_stats}"
    )


def client_label_stats(labels, client_indices):
    labels = np.asarray(labels)
    stats = []

    for cid, indices in enumerate(client_indices):
        if indices:
            unique, counts = np.unique(labels[indices], return_counts=True)
            label_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        else:
            label_counts = {}

        stats.append(
            {
                "client_id": cid,
                "num_samples": len(indices),
                "label_counts": label_counts,
            }
        )

    return stats
