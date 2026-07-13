"""
Unified Evaluation Pipeline for Recommendation Models.

Metrics:
  - Recall@K
  - NDCG@K  
  - Hit Rate@K
  - MRR (Mean Reciprocal Rank)

Supports:
  - Baseline models (Popularity, UserCF, ItemCF, BPR-MF)
  - Our proposed models (GNN, Sequence, Fusion, DeepFM)
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Callable


def recall_at_k(hits: np.ndarray, pos_count: int, k: int) -> float:
    """Recall@K = TP@K / total relevant items."""
    if pos_count == 0:
        return 0.0
    return hits[:k].sum() / pos_count


def ndcg_at_k(hits: np.ndarray, k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K."""
    dcg = 0.0
    for i in range(min(k, len(hits))):
        if hits[i] > 0:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG: best possible ranking (all positives at top)
    n_relevant = int(hits.sum())
    idcg = 0.0
    for i in range(min(k, n_relevant)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(hits: np.ndarray, k: int) -> float:
    """Hit Rate@K = 1 if at least one hit in top-K, else 0."""
    return 1.0 if hits[:k].sum() > 0 else 0.0


def mrr(ranked_items: List[int], positive_items: set) -> float:
    """Mean Reciprocal Rank: 1 / rank of first hit."""
    for rank, item in enumerate(ranked_items, start=1):
        if item in positive_items:
            return 1.0 / rank
    return 0.0


def evaluate_model(model, test_data, num_items: int, ks=[5, 10, 20]) -> Dict:
    """
    Evaluate a recommendation model on test data.
    
    Args:
        model: Object with recommend(user_idx, k) -> [(item_idx, score), ...]
        test_data: List of (user_idx, item_idx) test pairs
        num_items: Total number of items
        ks: List of K values for Recall@K, NDCG@K, Hit@K
    
    Returns:
        Dictionary of {metric_name: value}
    """
    # Build ground truth: user -> set of positive items
    user_positives = defaultdict(set)
    for u, i in test_data:
        user_positives[u].add(i)
    
    test_users = sorted(user_positives.keys())
    
    results = defaultdict(list)
    max_k = max(ks)
    
    for u in test_users:
        pos_set = user_positives[u]
        try:
            recs = model.recommend(u, k=max_k)
        except Exception:
            continue
        
        ranked_items = [item for item, _ in recs[:max_k]]
        n_pos = len(pos_set)
        
        # Build hits array
        hits = np.array([1.0 if item in pos_set else 0.0 for item in ranked_items])
        
        for k in ks:
            results[f'Recall@{k}'].append(recall_at_k(hits, n_pos, k))
            results[f'NDCG@{k}'].append(ndcg_at_k(hits, k))
            results[f'Hit@{k}'].append(hit_rate_at_k(hits, k))
        
        results['MRR'].append(mrr(ranked_items, pos_set))
    
    # Average across users
    summary = {}
    for metric, values in results.items():
        summary[metric] = np.mean(values)
    
    return summary


def evaluate_your_model(model_fn, model_name: str, data, ks=[5, 10, 20]) -> Dict:
    """
    Evaluate our proposed model (GNN, Sequence, Fusion, etc.)
    
    Args:
        model_fn: Function that takes user_idx and k, returns [(item_idx, score), ...]
        model_name: Name for logging
        data: Full data dict from data_export
        ks: List of K values
    """
    X_test = data['X_test']
    user_positives = defaultdict(set)
    for u, i in X_test:
        user_positives[u].add(i)
    
    test_users = sorted(user_positives.keys())
    results = defaultdict(list)
    max_k = max(ks)
    
    for u in test_users:
        pos_set = user_positives[u]
        try:
            recs = model_fn(u, k=max_k)
        except Exception:
            continue
        
        ranked_items = [item for item, _ in recs[:max_k]]
        n_pos = len(pos_set)
        
        hits = np.array([1.0 if item in pos_set else 0.0 for item in ranked_items])
        
        for k in ks:
            results[f'Recall@{k}'].append(recall_at_k(hits, n_pos, k))
            results[f'NDCG@{k}'].append(ndcg_at_k(hits, k))
            results[f'Hit@{k}'].append(hit_rate_at_k(hits, k))
        
        results['MRR'].append(mrr(ranked_items, pos_set))
    
    summary = {}
    for metric, values in results.items():
        summary[metric] = np.mean(values)
    
    return summary


def format_results_table(all_results: Dict[str, Dict]) -> str:
    """Format results as a LaTeX-compatible table."""
    metrics = ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 
               'Hit@10', 'MRR']
    model_names = list(all_results.keys())
    
    # Header
    lines = []
    header = "Model & " + " & ".join(metrics) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")
    
    # Rows
    for name in model_names:
        r = all_results[name]
        vals = []
        for m in metrics:
            v = r.get(m, 0.0)
            vals.append(f"{v:.4f}")
        
        # Bold the best value in each column
        row = f"{name} & " + " & ".join(vals) + " \\\\"
        lines.append(row)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)


def print_results(all_results: Dict[str, Dict]):
    """Pretty-print results."""
    metrics = ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 
               'Hit@10', 'MRR']
    
    # Find best values for highlighting
    best = {}
    for m in metrics:
        best[m] = max(r.get(m, 0.0) for r in all_results.values())
    
    print(f"\n{'='*90}")
    print(f"{'Model':<15}", end="")
    for m in metrics:
        print(f"{m:>10}", end="")
    print()
    print(f"{'-'*90}")
    
    for name in sorted(all_results.keys()):
        r = all_results[name]
        print(f"{name:<15}", end="")
        for m in metrics:
            v = r.get(m, 0.0)
            marker = " *" if v == best[m] else ""
            print(f"{v:>8.4f}{marker}", end="")
        print()
    
    print(f"{'='*90}")
    print("* = best in column")


if __name__ == '__main__':
    print("Evaluation module loaded. Use run_experiments.py to run full experiments.")
