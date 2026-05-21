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


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        neigh = adj @ x
        if self.aggr == 'mean':
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
            neigh = neigh / deg
        h_self = self.linear_self(x)
        h_neigh = self.linear_neigh(neigh)
        h = h_self + h_neigh
        h = self.norm(h)
        return F.relu(h)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = self.dropout(x)
        return self.output(x)


class HeteroGNNRecall(nn.Module):
    def __init__(self, num_users, num_problems, num_topics,
                 user_feat_dim=16, problem_feat_dim=16, topic_feat_dim=16,
                 hidden_dim=64, out_dim=64):
        super().__init__()
        self.user_proj = nn.Linear(user_feat_dim, hidden_dim)
        self.problem_proj = nn.Linear(problem_feat_dim, hidden_dim)
        self.topic_proj = nn.Linear(topic_feat_dim, hidden_dim)

        self.user_gnn = GraphSAGE(hidden_dim, hidden_dim, out_dim, num_layers=2)
        self.problem_gnn = GraphSAGE(hidden_dim, hidden_dim, out_dim, num_layers=2)
        self.topic_gnn = GraphSAGE(hidden_dim, hidden_dim, out_dim, num_layers=2)

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.problem_emb = nn.Embedding(num_problems, hidden_dim)
        self.topic_emb = nn.Embedding(num_topics, hidden_dim)

    def forward(self, user_adj, problem_adj, topic_adj,
                user_top_problem_adj, problem_topic_adj):

        u_init = self.user_emb.weight
        p_init = self.problem_emb.weight
        t_init = self.topic_emb.weight

        u_u = torch.mm(user_adj, u_init)
        u_p = torch.mm(user_top_problem_adj, p_init)
        u_agg = u_u + u_p

        p_u = torch.mm(user_top_problem_adj.t(), u_init)
        p_t = torch.mm(problem_topic_adj, t_init)
        p_agg = p_u + p_t

        t_p = torch.mm(problem_topic_adj.t(), p_init)
        t_agg = t_p

        u_emb = self.user_gnn(u_agg, user_adj)
        p_emb = self.problem_gnn(p_agg, problem_adj)
        t_emb = self.topic_gnn(t_agg, topic_adj)

        return u_emb, p_emb, t_emb


class GNNTrainer:
    def __init__(self, save_dir='recommend_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def export_graph_data(self):
        client = neo4j_client

        users = client.run_query("MATCH (u:User) RETURN u.user_id AS id ORDER BY id")
        problems = client.run_query("MATCH (p:Problem) RETURN p.problem_id AS id ORDER BY id")
        topics = client.run_query("MATCH (t:Topic) RETURN t.name AS name ORDER BY name")
        user_ids = [r['id'] for r in users]
        problem_ids = [r['id'] for r in problems]
        topic_names = [r['name'] for r in topics]
        num_users = len(user_ids)
        num_problems = len(problem_ids)
        num_topics = len(topic_names)
        user2idx = {uid: i for i, uid in enumerate(user_ids)}
        prob2idx = {pid: i for i, pid in enumerate(problem_ids)}
        topic2idx = {tn: i for i, tn in enumerate(topic_names)}
        idx2user = {i: uid for uid, i in user2idx.items()}
        idx2prob = {i: pid for pid, i in prob2idx.items()}
        idx2topic = {i: tn for tn, i in topic2idx.items()}

        user_adj = np.zeros((num_users, num_users))
        subs = client.run_query("""
            MATCH (u1:User)-[:SUBMITTED]->()-[:FOR]->(p:Problem)<-[:FOR]-()<-[:SUBMITTED]-(u2:User)
            WHERE u1 <> u2
            RETURN u1.user_id AS u1, u2.user_id AS u2, count(*) AS w
        """)
        for r in subs:
            i, j = user2idx.get(r['u1']), user2idx.get(r['u2'])
            if i is not None and j is not None:
                user_adj[i, j] = min(r['w'], 5.0)

        problem_adj = np.eye(num_problems)
        topic_adj = np.eye(num_topics)

        up_edges = client.run_query("""
            MATCH (u:User)-[:SUBMITTED]->()-[:FOR]->(p:Problem)
            RETURN u.user_id AS uid, p.problem_id AS pid, count(*) AS w
        """)
        up_adj = np.zeros((num_users, num_problems))
        for r in up_edges:
            i, j = user2idx.get(r['uid']), prob2idx.get(r['pid'])
            if i is not None and j is not None:
                up_adj[i, j] = min(r['w'], 5.0)

        pt_edges = client.run_query("""
            MATCH (p:Problem)-[:BELONGS_TO]->(t:Topic)
            RETURN p.problem_id AS pid, t.name AS tname
        """)
        pt_adj = np.zeros((num_problems, num_topics))
        for r in pt_edges:
            i, j = prob2idx.get(r['pid']), topic2idx.get(r['tname'])
            if i is not None and j is not None:
                pt_adj[i, j] = 1.0

        graph_data = {
            'user_adj': user_adj,
            'problem_adj': problem_adj,
            'topic_adj': topic_adj,
            'up_adj': up_adj,
            'pt_adj': pt_adj,
            'num_users': num_users,
            'num_problems': num_problems,
            'num_topics': num_topics,
            'user2idx': user2idx,
            'prob2idx': prob2idx,
            'topic2idx': topic2idx,
            'idx2user': idx2user,
            'idx2prob': idx2prob,
            'idx2topic': idx2topic,
        }

        with open(os.path.join(self.save_dir, 'graph_data.pkl'), 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"图数据导出完成: {num_users} 用户, {num_problems} 题目, {num_topics} 知识点")
        return graph_data

    def train(self, epochs=100, lr=0.001):
        data_path = os.path.join(self.save_dir, 'graph_data.pkl')
        if not os.path.exists(data_path):
            print("未找到图数据，开始导出...")
            self.export_graph_data()

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        user_adj = torch.tensor(data['user_adj'], dtype=torch.float)
        problem_adj = torch.tensor(data['problem_adj'], dtype=torch.float)
        topic_adj = torch.tensor(data['topic_adj'], dtype=torch.float)
        up_adj = torch.tensor(data['up_adj'], dtype=torch.float)
        pt_adj = torch.tensor(data['pt_adj'], dtype=torch.float)
        num_users = data['num_users']
        num_problems = data['num_problems']
        num_topics = data['num_topics']

        up_coo = up_adj.nonzero(as_tuple=False)
        pos_edges = up_coo
        num_neg_samples = min(pos_edges.size(0) * 2, 50000)
        all_users = torch.arange(num_users)
        all_probs = torch.arange(num_problems)
        neg_users = all_users[torch.randint(0, num_users, (num_neg_samples,))]
        neg_probs = all_probs[torch.randint(0, num_problems, (num_neg_samples,))]

        model = HeteroGNNRecall(num_users, num_problems, num_topics,
                                hidden_dim=64, out_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            u_emb, p_emb, t_emb = model(user_adj, problem_adj, topic_adj, up_adj, pt_adj)

            pos_u = u_emb[pos_edges[:, 0]]
            pos_p = p_emb[pos_edges[:, 1]]
            pos_scores = (pos_u * pos_p).sum(dim=1)
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()

            neg_u = u_emb[neg_users]
            neg_p = p_emb[neg_probs]
            neg_scores = (neg_u * neg_p).sum(dim=1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()

            reg_loss = 1e-4 * (u_emb.norm(2) + p_emb.norm(2) + t_emb.norm(2))
            loss = pos_loss + neg_loss + reg_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(pos_scores)
                    acc = ((pred > 0.5).float().mean().item())
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Pos Acc: {acc:.4f}")

        torch.save(model.state_dict(), os.path.join(self.save_dir, 'gnn_recall.pt'))
        emb_dict = {
            'user_emb': u_emb.detach().cpu().numpy(),
            'problem_emb': p_emb.detach().cpu().numpy(),
            'topic_emb': t_emb.detach().cpu().numpy(),
        }
        with open(os.path.join(self.save_dir, 'gnn_embeddings.pkl'), 'wb') as f:
            pickle.dump(emb_dict, f)
        print("GNN 模型与嵌入已保存")
        return model, emb_dict


def load_gnn_recall(model_dir='recommend_models'):
    emb_path = os.path.join(model_dir, 'gnn_embeddings.pkl')
    graph_path = os.path.join(model_dir, 'graph_data.pkl')
    if not os.path.exists(emb_path) or not os.path.exists(graph_path):
        return None, None

    with open(emb_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)

    return embeddings, graph_data


def gnn_recall(user_id, embeddings, graph_data, top_k=50):
    user_idx = graph_data['user2idx'].get(user_id)
    if user_idx is None:
        return []

    user_emb = embeddings['user_emb'][user_idx]
    problem_embs = embeddings['problem_emb']
    scores = np.dot(problem_embs, user_emb)

    up_adj = graph_data['up_adj']
    done_probs = set(np.where(up_adj[user_idx] > 0)[0].tolist())

    candidate_indices = np.argsort(scores)[::-1]
    results = []
    for idx in candidate_indices:
        if idx in done_probs:
            continue
        pid = graph_data['idx2prob'].get(idx)
        if pid is not None:
            results.append((pid, float(scores[idx])))
        if len(results) >= top_k:
            break

    return results


if __name__ == '__main__':
    trainer = GNNTrainer()
    trainer.export_graph_data()
    trainer.train(epochs=60)
