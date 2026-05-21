import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django
django.setup()

from utils.neo4j_client import neo4j_client


class RelGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, bias=True):
        super().__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.FloatTensor(num_relations, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.loop_weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.loop_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_list):
        out = torch.zeros(x.size(0), self.out_dim, device=x.device)

        for r in range(self.num_relations):
            if r < len(adj_list) and adj_list[r] is not None:
                support = torch.mm(x, self.weight[r])
                out += torch.mm(adj_list[r], support)

        out += torch.mm(x, self.loop_weight)

        if self.bias is not None:
            out += self.bias

        return F.relu(out)


class RGCNClassifier(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64, out_dim=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.layers = nn.ModuleList()
        self.layers.append(
            RelGraphConvLayer(hidden_dim, hidden_dim, num_relations)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                RelGraphConvLayer(hidden_dim, hidden_dim, num_relations)
            )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, adj_list, edge_index):
        x = self.embedding.weight

        for layer in self.layers:
            x = layer(x, adj_list)
            x = self.dropout(x)

        src_emb = x[edge_index[:, 0]]
        dst_emb = x[edge_index[:, 1]]
        cat = torch.cat([src_emb, dst_emb], dim=1)
        logits = self.output(cat).squeeze(-1)
        return torch.sigmoid(logits)

    def get_node_embeddings(self, adj_list):
        x = self.embedding.weight
        for layer in self.layers:
            x = layer(x, adj_list)
        return x


class RGCNDataBuilder:
    def __init__(self):
        self.topic2idx = {}
        self.idx2topic = {}
        self.relations = {}
        self.adj_list = []

    def build_from_neo4j(self):
        client = neo4j_client

        topics = client.run_query("MATCH (t:Topic) RETURN t.name AS name ORDER BY name")
        for i, r in enumerate(topics):
            self.topic2idx[r['name']] = i
            self.idx2topic[i] = r['name']

        num_topics = len(self.topic2idx)
        print(f"知识点数量: {num_topics}")

        self.relations = {
            'PREREQUISITE_OF': 0,
            'REVERSE_PREREQ': 1,
            'RELATED_TO': 2,
            'SHARED_PROBLEM': 3,
        }
        num_relations = len(self.relations)

        adj_mats = [None] * num_relations

        prereq = np.zeros((num_topics, num_topics))
        edges = client.run_query("""
            MATCH (t1:Topic)-[:PREREQUISITE_OF]->(t2:Topic)
            RETURN t1.name AS source, t2.name AS target
        """)
        for r in edges:
            i = self.topic2idx.get(r['source'])
            j = self.topic2idx.get(r['target'])
            if i is not None and j is not None:
                prereq[i, j] = 1.0

        reverse_prereq = prereq.T.copy()

        related = np.zeros((num_topics, num_topics))
        edges = client.run_query("""
            MATCH (t1:Topic)-[:RELATED_TO]->(t2:Topic)
            RETURN t1.name AS source, t2.name AS target
        """)
        for r in edges:
            i = self.topic2idx.get(r['source'])
            j = self.topic2idx.get(r['target'])
            if i is not None and j is not None:
                related[i, j] = 1.0

        shared = np.zeros((num_topics, num_topics))
        edges = client.run_query("""
            MATCH (t1:Topic)<-[:BELONGS_TO]-(p:Problem)-[:BELONGS_TO]->(t2:Topic)
            WHERE t1 <> t2
            RETURN t1.name AS t1, t2.name AS t2, count(p) AS cnt
        """)
        for r in edges:
            i = self.topic2idx.get(r['t1'])
            j = self.topic2idx.get(r['t2'])
            if i is not None and j is not None:
                shared[i, j] = min(r['cnt'], 5.0) / 5.0

        adj_mats[self.relations['PREREQUISITE_OF']] = prereq
        adj_mats[self.relations['REVERSE_PREREQ']] = reverse_prereq
        adj_mats[self.relations['RELATED_TO']] = related
        adj_mats[self.relations['SHARED_PROBLEM']] = shared

        for i in range(num_relations):
            if adj_mats[i] is not None:
                d = adj_mats[i].sum(axis=1, keepdims=True)
                d[d == 0] = 1
                adj_mats[i] = adj_mats[i] / d

        self.adj_list = adj_mats

        return num_topics, num_relations

    def build_edge_samples(self, num_neg_ratio=2):
        num_topics = len(self.topic2idx)

        pos_edges = []
        existing_pos = set()
        for i in range(num_topics):
            for j in range(num_topics):
                if self.adj_list[0][i, j] > 0:
                    pos_edges.append((i, j, 1))
                    existing_pos.add((i, j))

        num_pos = len(pos_edges)
        num_neg = min(num_pos * num_neg_ratio, num_topics * num_topics - num_pos)

        neg_edges = []
        neg_count = 0
        rng = np.random.RandomState(42)
        while neg_count < num_neg:
            i = rng.randint(0, num_topics)
            j = rng.randint(0, num_topics)
            if i != j and (i, j) not in existing_pos and (j, i) not in existing_pos:
                neg_edges.append((i, j, 0))
                existing_pos.add((i, j))
                neg_count += 1

        all_edges = np.array(pos_edges + neg_edges)
        rng.shuffle(all_edges)

        split = int(0.8 * len(all_edges))
        train_edges = all_edges[:split]
        test_edges = all_edges[split:]

        print(f"边样本: pos={num_pos}, neg={len(neg_edges)}, train={len(train_edges)}, test={len(test_edges)}")
        return train_edges, test_edges

    def save(self, path):
        adj_np = [np.array(a) if a is not None else None for a in self.adj_list]
        with open(path, 'wb') as f:
            pickle.dump({
                'topic2idx': self.topic2idx,
                'idx2topic': self.idx2topic,
                'relations': self.relations,
                'adj_list': adj_np,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.topic2idx = data['topic2idx']
            self.idx2topic = data['idx2topic']
            self.relations = data['relations']
            self.adj_list = data['adj_list']
        return self


class RGCNTrainer:
    def __init__(self, save_dir='recommend_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, epochs=100, lr=0.001):
        data_path = os.path.join(self.save_dir, 'rgcn_data.pkl')
        if os.path.exists(data_path):
            builder = RGCNDataBuilder().load(data_path)
            num_topics = len(builder.topic2idx)
            num_relations = len(builder.relations)
        else:
            builder = RGCNDataBuilder()
            num_topics, num_relations = builder.build_from_neo4j()
            builder.save(data_path)

        train_edges, test_edges = builder.build_edge_samples()

        adj_tensors = []
        for a in builder.adj_list:
            if a is not None:
                adj_tensors.append(torch.tensor(a, dtype=torch.float))
            else:
                adj_tensors.append(None)

        model = RGCNClassifier(num_topics, num_relations, hidden_dim=64,
                                num_layers=2, dropout=0.2)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_auc = 0
        for epoch in range(epochs):
            model.train()
            idx = np.random.permutation(len(train_edges))
            total_loss = 0
            batches = 0
            batch_size = 256

            for i in range(0, len(train_edges), batch_size):
                batch = train_edges[idx[i:i+batch_size]]
                edge_idx = torch.tensor(batch[:, :2], dtype=torch.long)
                labels = torch.tensor(batch[:, 2], dtype=torch.float)

                optimizer.zero_grad()
                pred = model(adj_tensors, edge_idx)
                loss = F.binary_cross_entropy(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_edge_idx = torch.tensor(test_edges[:, :2], dtype=torch.long)
                    test_labels = torch.tensor(test_edges[:, 2], dtype=torch.float)
                    pred_test = model(adj_tensors, test_edge_idx).numpy()
                    auc = self._compute_auc(test_labels.numpy(), pred_test)
                print(f"Epoch {epoch+1}, Loss: {total_loss/batches:.4f}, Test AUC: {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), os.path.join(self.save_dir, 'rgcn.pt'))

        torch.save(model.state_dict(), os.path.join(self.save_dir, 'rgcn.pt'))
        print(f"RGCN 模型已保存, Best AUC: {best_auc:.4f}")

        model.eval()
        with torch.no_grad():
            node_embs = model.get_node_embeddings(adj_tensors).cpu().numpy()
        emb_dict = {
            'topic2idx': builder.topic2idx,
            'idx2topic': builder.idx2topic,
            'embeddings': node_embs,
        }
        with open(os.path.join(self.save_dir, 'rgcn_embeddings.pkl'), 'wb') as f:
            pickle.dump(emb_dict, f)

        return model, builder

    def _compute_auc(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)
        if pos == 0 or neg == 0:
            return 0.5
        tp, auc_val = 0, 0
        for idx in order:
            if y_true[idx] == 1:
                tp += 1
            else:
                auc_val += tp
        return auc_val / (pos * neg)

    def predict_prerequisites(self, confidence_threshold=0.7, top_k_per_node=5):
        model_path = os.path.join(self.save_dir, 'rgcn.pt')
        data_path = os.path.join(self.save_dir, 'rgcn_data.pkl')

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print("未找到模型，请先训练")
            return []

        builder = RGCNDataBuilder().load(data_path)
        num_topics = len(builder.topic2idx)
        num_relations = len(builder.relations)

        adj_tensors = []
        for a in builder.adj_list:
            if a is not None:
                adj_tensors.append(torch.tensor(a, dtype=torch.float))
            else:
                adj_tensors.append(None)

        model = RGCNClassifier(num_topics, num_relations)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        existing_edges = set()
        prereq_mat = builder.adj_list[0]
        for i in range(num_topics):
            for j in range(num_topics):
                if prereq_mat[i, j] > 0:
                    existing_edges.add((i, j))

        candidates = []
        with torch.no_grad():
            for i in range(num_topics):
                candidate_js = list(range(num_topics))
                candidate_js = [j for j in candidate_js if i != j and (i, j) not in existing_edges]

                if not candidate_js:
                    continue

                edge_idx = torch.tensor([[i, j] for j in candidate_js], dtype=torch.long)
                scores = model(adj_tensors, edge_idx).numpy()

                for k, j in enumerate(candidate_js):
                    if scores[k] > confidence_threshold:
                        candidates.append((i, j, float(scores[k])))

        candidates.sort(key=lambda x: x[2], reverse=True)

        results = []
        seen_pairs = set()
        for i, j, score in candidates:
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            results.append({
                'source': builder.idx2topic[i],
                'target': builder.idx2topic[j],
                'confidence': round(score, 4),
            })
            if len(results) >= 30:
                break

        return results


def load_rgcn(model_dir='recommend_models'):
    model_path = os.path.join(model_dir, 'rgcn.pt')
    data_path = os.path.join(model_dir, 'rgcn_data.pkl')
    emb_path = os.path.join(model_dir, 'rgcn_embeddings.pkl')

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None, None

    builder = RGCNDataBuilder().load(data_path)
    num_topics = len(builder.topic2idx)
    num_relations = len(builder.relations)

    adj_tensors = []
    for a in builder.adj_list:
        if a is not None:
            adj_tensors.append(torch.tensor(a, dtype=torch.float))
        else:
            adj_tensors.append(None)

    model = RGCNClassifier(num_topics, num_relations)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    embeddings = None
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            embeddings = pickle.load(f)

    return model, builder, adj_tensors


if __name__ == '__main__':
    trainer = RGCNTrainer()
    trainer.train(epochs=100)

    print("\n=== RGCN 预测的 PREREQUISITE_OF 关系 ===")
    predictions = trainer.predict_prerequisites(confidence_threshold=0.5)
    for p in predictions:
        print(f"  {p['source']} → {p['target']} (confidence={p['confidence']:.4f})")
