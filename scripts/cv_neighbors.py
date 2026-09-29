"""Exact cached Hamming distances for repeated fold-local kNN evaluation."""

import numpy as np
from scipy.sparse import csr_matrix


def make_distance_cache(encoded, path, chunk_size=256):
    """Store mismatch counts, not learned features; no labels are consulted.

    Subtracting one reference one-hot sequence makes the matrix sparse while
    preserving Euclidean distances: squared one-hot distance = 2 * mismatches.
    """
    n, length = encoded.shape
    if length > 65535:
        raise ValueError('Sequence length exceeds cache integer range')
    reference = encoded[0]
    rows, positions = np.where(encoded != reference)
    columns_mutant = positions*20 + encoded[rows, positions]
    columns_reference = positions*20 + reference[positions]
    delta = csr_matrix((np.concatenate([np.ones(len(rows), dtype=np.int32), -np.ones(len(rows), dtype=np.int32)]),
                        (np.concatenate([rows, rows]), np.concatenate([columns_mutant, columns_reference]))),
                       shape=(n, length*20), dtype=np.int32)
    norms = np.asarray(delta.multiply(delta).sum(axis=1)).ravel()
    cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8 if length <= 255 else np.uint16, shape=(n, n))
    for start in range(0, n, chunk_size):
        end = min(n, start+chunk_size)
        distances = (norms[start:end, None]+norms[None, :] - 2*(delta[start:end] @ delta.T).toarray())//2
        if np.any(distances < 0) or np.any(distances > length):
            raise ValueError('Invalid Hamming distance')
        cache[start:end] = distances
    cache.flush()
    return cache


def query_cache(cache, query_ids, train_ids, k, length, batch_size=256):
    """Match sklearn 1.8 brute-Hamming argpartition and boundary-tie behavior."""
    if not 1 <= k <= len(train_ids):
        raise ValueError('Invalid k')
    distances, indices = [], []
    for start in range(0, len(query_ids), batch_size):
        query = query_ids[start:start+batch_size]
        block = np.asarray(cache[np.ix_(query, train_ids)], dtype=np.float64)/length
        nearest = np.argpartition(block, k-1, axis=1)[:, :k]
        sample = np.arange(len(query))[:, None]
        nearest = nearest[sample, np.argsort(block[sample, nearest], axis=1)]
        indices.append(nearest)
        distances.append(block[sample, nearest])
    return np.concatenate(distances), np.concatenate(indices)
