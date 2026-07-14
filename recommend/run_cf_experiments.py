"""
Run recommendation experiments on Codeforces data.

Loads pickled data from /data/codeforces/data.pkl,
runs baseline models + our KG-based method,
outputs LaTeX table.

Usage (in container):
    python3 /app/recommend/run_cf_experiments.py
"""

import os
import sys
import pickle
import logging
import numpy as np
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = '/home/proach/publicData/processed/codeforces_u2000_min5.pkl'


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def recall_at_k(recs, true_items, k):
    if not true_items:
        return 0.0
    hits = len(set(r for r, _ in recs[:k]) & set(true_items))
    return hits / min(k, len(true_items))


def ndcg_at_k(recs, true_items, k):
    if not true_items:
        return 0.0
    dcg = 0.0
    for i, (item, _) in enumerate(recs[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(recs, true_items, k):
    if not true_items:
        return 0.0
    return 1.0 if any(r in true_items for r, _ in recs[:k]) else 0.0


def mrr(recs, true_items):
    if not true_items:
        return 0.0
    for i, (item, _) in enumerate(recs):
        if item in true_items:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(recommend_fn, X_test, ks=[5, 10, 20]):
    """recommend_fn(user_idx, k) → [(item_idx, score), ...]"""
    # Group test items by user
    user_test = defaultdict(set)
    for u, p in X_test:
        user_test[u].add(p)
    all_users = sorted(user_test.keys())
    metrics = {f'Recall@{k}': [] for k in ks}
    metrics.update({f'NDCG@{k}': [] for k in ks})
    metrics.update({f'Hit@{k}': [] for k in ks})
    metrics['MRR'] = []
    for u in all_users:
        true = user_test[u]
        recs = recommend_fn(u, max(ks))
        if not recs:
            for k in ks:
                metrics[f'Recall@{k}'].append(0.0)
                metrics[f'NDCG@{k}'].append(0.0)
                metrics[f'Hit@{k}'].append(0.0)
            metrics['MRR'].append(0.0)
            continue
        for k in ks:
            metrics[f'Recall@{k}'].append(recall_at_k(recs, true, k))
            metrics[f'NDCG@{k}'].append(ndcg_at_k(recs, true, k))
            metrics[f'Hit@{k}'].append(hit_at_k(recs, true, k))
        metrics['MRR'].append(mrr(recs, true))
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────
# Baseline Models
# ──────────────────────────────────────────────────────────────────────

class Popularity:
    def __init__(self, n_items):
        self.n_items = n_items
        self.sorted_items = None

    def fit(self, X_train):
        counts = defaultdict(int)
        for u, p in X_train:
            counts[p] += 1
        self.sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        logger.info(f"  Popularity: {len(self.sorted_items)} items, top counts: {self.sorted_items[:3]}")

    def recommend(self, user_idx, k=20):
        return [(p, 1.0) for p, _ in self.sorted_items[:k]]


class UserCF:
    def __init__(self, n_users, n_items, K=50):
        self.n_users = n_users
        self.n_items = n_items
        self.K = K

    def fit(self, X_train):
        logger.info(f"  UserCF: building {self.n_users}x{self.n_items} matrix...")
        self.user_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for u, p in X_train:
            self.user_matrix[u, p] = 1.0
        # Cosine similarity
        norms = np.linalg.norm(self.user_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.user_matrix_normed = self.user_matrix / norms
        self.similarity = self.user_matrix_normed @ self.user_matrix_normed.T
        np.fill_diagonal(self.similarity, 0)
        self.pred = self.similarity @ self.user_matrix

    def recommend(self, user_idx, k=20):
        if user_idx >= self.n_users:
            return []
        scores = self.pred[user_idx].copy()
        scores[self.user_matrix[user_idx] > 0] = -1  # Exclude interacted
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > -1][:k]


class ItemCF:
    def __init__(self, n_users, n_items, K=50):
        self.n_users = n_users
        self.n_items = n_items
        self.K = K

    def fit(self, X_train):
        logger.info(f"  ItemCF: building {self.n_users}x{self.n_items} matrix...")
        self.item_matrix = np.zeros((self.n_items, self.n_users), dtype=np.float32)
        for u, p in X_train:
            self.item_matrix[p, u] = 1.0
        norms = np.linalg.norm(self.item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.item_matrix_normed = self.item_matrix / norms
        # item similarity: (n_items, n_items)
        self.similarity = self.item_matrix_normed @ self.item_matrix_normed.T
        np.fill_diagonal(self.similarity, 0)
        self.pred = self.item_matrix.T @ self.similarity  # (n_users, n_items)

    def recommend(self, user_idx, k=20):
        if user_idx >= self.n_users:
            return []
        scores = self.pred[user_idx].copy()
        # Exclude interacted items
        interacted = np.where(self.item_matrix[:, user_idx] > 0)[0]
        scores[interacted] = -1
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > -1][:k]


class BPRMF:
    def __init__(self, n_users, n_items, dim=64, lr=0.01, reg=0.01, epochs=20, batch_size=2048, n_samples=200000):
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_samples = n_samples

    def fit(self, X_train):
        import torch
        logger.info(f"  BPR-MF: dim={self.dim}, epochs={self.epochs}, batch={self.batch_size}, samples={self.n_samples}")
        
        # Subsample training data
        n = min(len(X_train), self.n_samples)
        idx = np.random.choice(len(X_train), n, replace=False)
        X_sample = [X_train[i] for i in idx]
        
        U = torch.nn.Embedding(self.n_users, self.dim)
        I = torch.nn.Embedding(self.n_items, self.dim)
        torch.nn.init.normal_(U.weight, std=0.1)
        torch.nn.init.normal_(I.weight, std=0.1)
        opt = torch.optim.Adam(list(U.parameters()) + list(I.parameters()), lr=self.lr)
        
        user_items = defaultdict(set)
        for u, p in X_sample:
            user_items[u].add(p)
        
        n_batches = max(1, n // self.batch_size)
        for ep in range(self.epochs):
            total_loss = 0.0
            perm = np.random.permutation(n)
            for b in range(n_batches):
                batch_idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                if len(batch_idx) == 0:
                    continue
                users = [X_sample[i][0] for i in batch_idx]
                pos_items = [X_sample[i][1] for i in batch_idx]
                neg_items = []
                for u in users:
                    while True:
                        ni = np.random.randint(0, self.n_items)
                        if ni not in user_items[u]:
                            neg_items.append(ni)
                            break
                
                u_t = torch.tensor(users, dtype=torch.long)
                p_t = torch.tensor(pos_items, dtype=torch.long)
                n_t = torch.tensor(neg_items, dtype=torch.long)
                
                u_emb = U(u_t)
                p_emb = I(p_t)
                n_emb = I(n_t)
                
                pos = (u_emb * p_emb).sum(dim=1)
                neg = (u_emb * n_emb).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
                loss += self.reg * (u_emb.pow(2).sum() + p_emb.pow(2).sum() + n_emb.pow(2).sum()) / len(batch_idx)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            if (ep + 1) % 5 == 0:
                logger.info(f"    Epoch {ep+1}/{self.epochs}, loss={total_loss/n_batches:.4f}")

        self.U = U.weight.detach().numpy()
        self.I = I.weight.detach().numpy()
        self.user_items = user_items

    def recommend(self, user_idx, k=20):
        if user_idx >= self.n_users:
            return []
        u_vec = self.U[user_idx]
        scores = u_vec @ self.I.T
        interacted = self.user_items.get(user_idx, set())
        scores[list(interacted)] = -1e10
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > -1e9][:k]


# ──────────────────────────────────────────────────────────────────────
# KG-based Recommender
# ──────────────────────────────────────────────────────────────────────

class KGRecommender:
    """
    Knowledge Graph recommender using Codeforces tags + co-occurrence.
    
    Strategy:
    1. Build user topic profile from solved problems
    2. Expand profile via tag co-occurrence (similar tags → relevance boost)
    3. Score candidate problems by weighted tag overlap
    4. Boost unsolved topics slightly for diversity
    """
    def __init__(self, data):
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.prob_tags = data['prob_tags']
        self.id2prob = data['id2prob']
        self.tag2id = data['tag2id']
        self.all_tags = data['all_tags']
        
        # Tag co-occurrence matrix (38x38)
        n_tags = len(self.all_tags)
        tag_cooccur = data['tag_cooccur']
        self.tag_sim = np.zeros((n_tags, n_tags))
        for (a, b), cnt in tag_cooccur.items():
            if a in self.tag2id and b in self.tag2id:
                self.tag_sim[self.tag2id[a], self.tag2id[b]] = cnt
                self.tag_sim[self.tag2id[b], self.tag2id[a]] = cnt
        # Normalize rows to sum=1 (probability distribution)
        row_sums = self.tag_sim.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.tag_sim = self.tag_sim / row_sums
        
        # Problem → tag vector (n_items × n_tags)
        self.prob_tag_mat = np.zeros((self.n_items, n_tags))
        for p_idx in range(self.n_items):
            pid = self.id2prob.get(p_idx, '')
            for t in data['prob_tags'].get(pid, []):
                if t in self.tag2id:
                    self.prob_tag_mat[p_idx, self.tag2id[t]] = 1.0

    def fit(self, X_train):
        # User → tag vector
        self.user_tags = np.zeros((self.n_users, len(self.all_tags)))
        for u, p in X_train:
            self.user_tags[u] += self.prob_tag_mat[p]
        # Normalize per user
        row_sums = self.user_tags.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.user_tags_norm = self.user_tags / row_sums
        
        # Expanded tag profile (tag propagation: user_tags × tag_sim)
        self.user_tags_expanded = self.user_tags_norm @ self.tag_sim

    def recommend(self, user_idx, k=20):
        if user_idx >= self.n_users:
            return []
        user_vec = 0.6 * self.user_tags_norm[user_idx] + 0.4 * self.user_tags_expanded[user_idx]
        # Score = cosine similarity between user tag vector and problem tag vector
        scores = self.prob_tag_mat @ user_vec  # (n_items,)
        # Penalize already-solved problems
        solved = np.where(self.user_tags[user_idx] > 0.5)[0]  # tags with significant presence
        if len(solved) > 0:
            # Find problems that heavily overlap with solved tags
            overlap = self.prob_tag_mat[:, solved].sum(axis=1)
            mask = overlap >= len(solved) * 0.8  # 80%+ overlap = essentially already mastered
            scores[mask] *= 0.3  # down-weight but don't eliminate
        
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        result = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
        return result[:k]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"  Users: {data['n_users']}, Items: {data['n_items']}")
    logger.info(f"  Train: {data['n_train']}, Test: {data['n_test']}")
    logger.info(f"  Tags: {len(data['all_tags'])}")

    X_train = data['X_train']
    X_test = data['X_test']
    n_users = data['n_users']
    n_items = data['n_items']

    results = {}
    ks = [5, 10, 20]

    # 1. Popularity
    logger.info("=== Popularity ===")
    t0 = time.time()
    pop = Popularity(n_items)
    pop.fit(X_train)
    results['Popularity'] = evaluate(pop.recommend, X_test, ks)
    logger.info(f"  {results['Popularity']['Recall@10']}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")

    # 2. UserCF
    logger.info("=== UserCF ===")
    t0 = time.time()
    ucf = UserCF(n_users, n_items)
    ucf.fit(X_train)
    results['UserCF'] = evaluate(ucf.recommend, X_test, ks)
    logger.info(f"  {results['UserCF']['Recall@10']}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")

    # 3. ItemCF
    logger.info("=== ItemCF ===")
    t0 = time.time()
    icf = ItemCF(n_users, n_items)
    icf.fit(X_train)
    results['ItemCF'] = evaluate(icf.recommend, X_test, ks)
    logger.info(f"  {results['ItemCF']['Recall@10']}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")

    # 4. BPR-MF (requires torch - skip if not available)
    logger.info("=== BPR-MF ===")
    try:
        import torch
        t0 = time.time()
        bpr = BPRMF(n_users, n_items, dim=64, epochs=30)
        bpr.fit(X_train)
        results['BPR-MF'] = evaluate(bpr.recommend, X_test, ks)
        logger.info(f"  {results['BPR-MF']['Recall@10']}")
        logger.info(f"  Time: {time.time() - t0:.1f}s")
    except ImportError:
        logger.info("  Skipped (torch not available)")

    # 5. KG Recommender
    logger.info("=== KG-Rec ===")
    t0 = time.time()
    kg = KGRecommender(data)
    kg.fit(X_train)
    results['KG-Rec'] = evaluate(kg.recommend, X_test, ks)
    logger.info(f"  {results['KG-Rec']['Recall@10']}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")

    # Print table
    print("\n" + "=" * 70)
    print("Results Table")
    print("=" * 70)
    header = f"{'Model':<14} {'R@5':>12} {'R@10':>12} {'R@20':>12} {'N@10':>12} {'H@10':>12} {'MRR':>12}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        line = f"{name:<14}"
        for k in ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'Hit@10']:
            mu, std = m[k]
            line += f" {mu:>11.4f}"
        mu, std = m['MRR']
        line += f" {mu:>11.4f}"
        print(line)
    print("=" * 70)


if __name__ == '__main__':
    main()
