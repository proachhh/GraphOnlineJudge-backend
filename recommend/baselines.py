"""
Baseline Models for Recommendation Evaluation
- Popularity
- UserCF (User-based Collaborative Filtering)
- ItemCF (Item-based Collaborative Filtering)
- BPR-MF (Bayesian Personalized Ranking with Matrix Factorization)

All models follow the same interface:
  __init__(num_users, num_items)
  fit(train_pairs)   # train_pairs: list of (user_idx, item_idx)
  predict(user_idx, item_idx) -> float score
  recommend(user_idx, k=10) -> list of (item_idx, score)
"""

import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
import pickle
import os


class Popularity:
    """Most popular items baseline."""
    
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.item_popularity = np.zeros(num_items)
        self.sorted_items = None
    
    def fit(self, train_pairs):
        for _, item_idx in train_pairs:
            self.item_popularity[item_idx] += 1
        self.sorted_items = np.argsort(-self.item_popularity)
    
    def predict(self, user_idx, item_idx):
        return float(self.item_popularity[item_idx])
    
    def recommend(self, user_idx, k=10):
        return [(int(idx), float(self.item_popularity[idx])) 
                for idx in self.sorted_items[:k]]


class UserCF:
    """User-based Collaborative Filtering with cosine similarity."""
    
    def __init__(self, num_users, num_items, K=50):
        self.num_users = num_users
        self.num_items = num_items
        self.K = K
        self.user_item_matrix = None
        self.user_similarity = None
    
    def fit(self, train_pairs):
        # Build sparse user-item matrix
        rows, cols = zip(*train_pairs) if train_pairs else ([], [])
        data = np.ones(len(rows))
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(self.num_users, self.num_items)
        )
        self._compute_similarity()
    
    def _compute_similarity(self):
        """Pre-compute top-K similar users."""
        from sklearn.metrics.pairwise import cosine_similarity
        dense = self.user_item_matrix.toarray()
        self.user_similarity = cosine_similarity(dense)
    
    def predict(self, user_idx, item_idx):
        if user_idx >= self.num_users:
            return 0.0
        sims = self.user_similarity[user_idx]
        top_k = np.argsort(-sims)[1:self.K+1]  # exclude self
        num = denom = 0.0
        for v in top_k:
            if sims[v] <= 0:
                continue
            r_vi = self.user_item_matrix[v, item_idx]
            num += sims[v] * r_vi
            denom += sims[v]
        return num / denom if denom > 0 else 0.0
    
    def recommend(self, user_idx, k=10):
        scores = np.array([self.predict(user_idx, i) for i in range(self.num_items)])
        top_k = np.argsort(-scores)[:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]


class ItemCF:
    """Item-based Collaborative Filtering with cosine similarity."""
    
    def __init__(self, num_users, num_items, K=50):
        self.num_users = num_users
        self.num_items = num_items
        self.K = K
        self.user_item_matrix = None
        self.item_similarity = None
    
    def fit(self, train_pairs):
        rows, cols = zip(*train_pairs) if train_pairs else ([], [])
        data = np.ones(len(rows))
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(self.num_users, self.num_items)
        )
        self._compute_similarity()
    
    def _compute_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity
        item_matrix = self.user_item_matrix.T.toarray()
        norms = np.linalg.norm(item_matrix, axis=0, keepdims=True)
        norms[norms == 0] = 1
        item_matrix = item_matrix / norms
        self.item_similarity = item_matrix @ item_matrix.T
    
    def predict(self, user_idx, item_idx):
        if item_idx >= self.num_items:
            return 0.0
        user_items = self.user_item_matrix[user_idx].toarray().flatten()
        interacted = np.where(user_items > 0)[0]
        if len(interacted) == 0:
            return 0.0
        sims = self.item_similarity[item_idx, interacted]
        top_k = np.argsort(-sims)[:self.K]
        num = sims[top_k].sum()
        denom = np.abs(sims[top_k]).sum()
        return float(num / denom) if denom > 0 else 0.0
    
    def recommend(self, user_idx, k=10):
        user_items = self.user_item_matrix[user_idx].toarray().flatten()
        scores = np.zeros(self.num_items)
        interacted = np.where(user_items > 0)[0]
        if len(interacted) > 0:
            for j in interacted:
                top_sim_items = np.argsort(-self.item_similarity[j])[:self.K]
                for item in top_sim_items:
                    if user_items[item] == 0:
                        scores[item] += self.item_similarity[j, item]
        top_k = np.argsort(-scores)[:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]


class BPRMF:
    """
    Bayesian Personalized Ranking with Matrix Factorization.
    Uses PyTorch for training.
    """
    
    def __init__(self, num_users, num_items, embed_dim=64, lr=0.01, epochs=50, 
                 batch_size=512, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        
        self.user_emb = None
        self.item_emb = None
        self.user_pos_items = defaultdict(set)
    
    def fit(self, train_pairs):
        import torch
        
        # Build user positive items index
        for u, i in train_pairs:
            self.user_pos_items[u].add(i)
        
        users = np.array([u for u, _ in train_pairs])
        pos_items = np.array([i for _, i in train_pairs])
        all_items = set(range(self.num_items))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        U = torch.nn.Embedding(self.num_users, self.embed_dim)
        I = torch.nn.Embedding(self.num_items, self.embed_dim)
        torch.nn.init.normal_(U.weight, std=0.1)
        torch.nn.init.normal_(I.weight, std=0.1)
        U = U.to(device)
        I = I.to(device)
        
        optimizer = torch.optim.Adam(list(U.parameters()) + list(I.parameters()), 
                                      lr=self.lr, weight_decay=self.reg)
        
        n_pairs = len(users)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_pairs)
            total_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_pairs, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                u_batch = torch.tensor(users[batch_idx], dtype=torch.long, device=device)
                i_batch = torch.tensor(pos_items[batch_idx], dtype=torch.long, device=device)
                
                # Negative sampling: for each positive item, sample one negative
                neg_items = []
                for u_idx in u_batch.cpu().numpy():
                    user_neg_pool = list(all_items - self.user_pos_items[u_idx])
                    if user_neg_pool:
                        neg_items.append(np.random.choice(user_neg_pool))
                    else:
                        neg_items.append(0)
                j_batch = torch.tensor(neg_items, dtype=torch.long, device=device)
                
                u_emb = U(u_batch)  # [B, D]
                i_emb = I(i_batch)  # [B, D]
                j_emb = I(j_batch)  # [B, D]
                
                x_ui = (u_emb * i_emb).sum(dim=1)  # [B]
                x_uj = (u_emb * j_emb).sum(dim=1)  # [B]
                
                loss = -torch.log(torch.sigmoid(x_ui - x_uj) + 1e-10).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"  BPR-MF Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.user_emb = U.weight.detach().cpu().numpy()
        self.item_emb = I.weight.detach().cpu().numpy()
        return self
    
    def predict(self, user_idx, item_idx):
        if self.user_emb is None:
            return 0.0
        return float(np.dot(self.user_emb[user_idx], self.item_emb[item_idx]))
    
    def recommend(self, user_idx, k=10):
        if self.user_emb is None:
            return []
        scores = self.user_emb[user_idx] @ self.item_emb.T
        top_k = np.argsort(-scores)[:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]


def load_data():
    """Load the exported recommendation data."""
    data_path = '/data/recommend_data.pkl'
    if not os.path.exists(data_path):
        print("Data not found, running data export...")
        import recommend.data_export
        recommend.data_export.export()
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_baselines(data):
    """Train all baseline models and return them."""
    X_train = data['X_train']
    num_users = data['num_users']
    num_items = data['num_items']
    
    print(f"Training baselines on {len(X_train)} pairs, {num_users} users, {num_items} items")
    
    models = {}
    
    # 1. Popularity
    print("\n[1/4] Training Popularity baseline...")
    pop = Popularity(num_users, num_items)
    pop.fit(X_train)
    models['Popularity'] = pop
    print("  Done.")
    
    # 2. UserCF
    print("\n[2/4] Training UserCF baseline...")
    ucf = UserCF(num_users, num_items, K=50)
    ucf.fit(X_train)
    models['UserCF'] = ucf
    print("  Done.")
    
    # 3. ItemCF
    print("\n[3/4] Training ItemCF baseline...")
    icf = ItemCF(num_users, num_items, K=50)
    icf.fit(X_train)
    models['ItemCF'] = icf
    print("  Done.")
    
    # 4. BPR-MF
    print("\n[4/4] Training BPR-MF baseline...")
    bpr = BPRMF(num_users, num_items, embed_dim=64, lr=0.01, epochs=50)
    bpr.fit(X_train)
    models['BPR-MF'] = bpr
    print("  Done.")
    
    return models


if __name__ == '__main__':
    data = load_data()
    models = train_baselines(data)
    print("\nAll baselines trained successfully!")
