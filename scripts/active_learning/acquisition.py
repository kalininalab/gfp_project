"""Jannis's acquisition rule, independent of models, labels and file paths.

Inputs are model-derived arrays and amino-acid strings. Unmeasured targets are
deliberately absent from this API. Returned indices address the supplied pool.
"""
import numpy as np
import torch
from sklearn.cluster import SpectralClustering


def acquisition_scores(labeled_embeddings, pool_embeddings, variance, alpha=0.38):
    """Mix scaled distance and MC variance using the manuscript equation.

    Embeddings have shape (samples, features); variance has shape (pool_size,).
    Distances are computed in CPU chunks to bound temporary memory. Constant
    uncertainty (including a model without dropout) contributes exactly zero.
    """
    labeled = torch.as_tensor(labeled_embeddings, dtype=torch.float32)
    pool = torch.as_tensor(pool_embeddings, dtype=torch.float32)
    variance = np.asarray(variance).reshape(-1)
    if len(labeled) == 0 or len(pool) == 0 or len(variance) != len(pool):
        raise ValueError('Nonempty labeled/pool embeddings and matching variance required')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1]')
    if not all(np.isfinite(x).all() for x in [labeled.numpy(), pool.numpy(), variance]):
        raise ValueError('Nonfinite acquisition inputs')
    distances = torch.cat([
        torch.cdist(chunk, labeled).min(dim=1).values
        for chunk in pool.split(1024)
    ]).numpy()

    def scale(values):
        return (values - values.min()) / (np.ptp(values) + 1e-8)

    return alpha * scale(distances) + (1 - alpha) * scale(variance)


def diverse_sequences(sequences, count, kmer_size=3):
    """Greedy farthest-first selection using the student's 3-mer distance.

    The first input is selected first; therefore callers pass descending-score
    candidates. The reference denominator uses the number of 3-mer positions,
    not the cardinality of the set union. This intentionally preserves its
    behavior even for repeated 3-mers; it is not exact Jaccard distance.
    """
    if not 0 <= count <= len(sequences):
        raise ValueError('Invalid diversity count')
    if count == 0:
        return []
    length = len(sequences[0])
    if length < kmer_size or any(len(s) != length for s in sequences):
        raise ValueError('Equal-length sequences of at least kmer_size required')
    sets = [{s[i:i+kmer_size] for i in range(length-kmer_size+1)} for s in sequences]
    selected = [0]
    distances = np.full(len(sequences), np.inf)
    while len(selected) < count:
        last = sets[selected[-1]]
        for i, candidate in enumerate(sets):
            intersection = len(last & candidate)
            distance = 1 - intersection / (2*(length-kmer_size+1)-intersection)
            distances[i] = min(distances[i], distance)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return selected


def cluster_budgets(assignments, budget):
    """Allocate one query per cluster, then proportionally by pool membership.

    Largest remainders receive leftover slots. Capacity checks fix the reference
    implementation's underfilling when a cluster has fewer candidates than its
    allocation. Returns (sorted_cluster_labels, integer_budgets).
    """
    labels, counts = np.unique(assignments, return_counts=True)
    if budget < 0 or budget > counts.sum():
        raise ValueError('Query budget exceeds available pool')
    allocation = np.zeros(len(labels), dtype=int)
    if budget >= len(labels):
        allocation[:] = 1
    remaining = budget - allocation.sum()
    exact = counts / counts.sum() * remaining
    allocation += np.floor(exact).astype(int)
    order = np.argsort(exact - np.floor(exact))[::-1]
    allocation = np.minimum(allocation, counts)
    while allocation.sum() < budget:
        for i in order:
            if allocation[i] < counts[i]:
                allocation[i] += 1
                if allocation.sum() == budget:
                    break
    return labels, allocation


def select_queries(scores, pool_embeddings, sequences, budget):
    """Select pool-local indices with the original two-cluster RBF algorithm.

    Within each cluster, shortlist up to 5 times its budget by acquisition score,
    then apply sequence diversity. SpectralClustering uses random_state=0,
    gamma=1 and dense RBF affinity, as in the reference. Memory is quadratic in
    pool size: the default experiment has about 2,452 pool members.
    """
    scores = np.asarray(scores)
    if len(scores) != len(sequences) or len(scores) != len(pool_embeddings):
        raise ValueError('Pool inputs must have matching lengths')
    if not 0 < budget <= len(scores):
        raise ValueError('Invalid query budget')
    assignments = (SpectralClustering(n_clusters=2, random_state=0,
                    affinity='rbf', gamma=1.0).fit_predict(pool_embeddings)
                   if len(scores) > 2 else np.zeros(len(scores), dtype=int))
    labels, budgets = cluster_budgets(assignments, budget)
    selected = []
    for label, count in zip(labels, budgets):
        local = np.flatnonzero(assignments == label)
        candidates = local[np.argsort(-scores[local])[:min(len(local), count*5)]]
        chosen = diverse_sequences([sequences[i] for i in candidates], int(count))
        selected.extend(candidates[chosen].tolist())
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError('Acquisition failed to fill its budget uniquely')
    return np.asarray(selected, dtype=int)
