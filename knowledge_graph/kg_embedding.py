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
from problem.models import ProblemTag


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=128, margin=1.0, p_norm=2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.margin = margin
        self.p_norm = p_norm

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        nn.init.normal_(self.entity_emb.weight, 0, 1.0 / np.sqrt(embed_dim))
        nn.init.normal_(self.relation_emb.weight, 0, 1.0 / np.sqrt(embed_dim))

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        score = torch.norm(h + r - t, p=self.p_norm, dim=1)
        return score

    def score_triple(self, head, relation, tail):
        with torch.no_grad():
            h = self.entity_emb(head)
            r = self.relation_emb(relation)
            t = self.entity_emb(tail)
            return -torch.norm(h + r - t, p=self.p_norm, dim=1)


class KnowledgeGraphData:
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.id2entity = {}
        self.id2relation = {}
        self.triples = []

    def build_from_neo4j(self):
        client = neo4j_client

        problems = client.run_query("MATCH (p:Problem) RETURN p.problem_id AS id")
        topics = client.run_query("MATCH (t:Topic) RETURN t.name AS name")

        for i, r in enumerate(problems):
            eid = f"problem:{r['id']}"
            self.entities[eid] = len(self.entities)
        for r in topics:
            eid = f"topic:{r['name']}"
            self.entities[eid] = len(self.entities)

        self.relations['BELONGS_TO'] = len(self.relations)
        self.relations['PREREQUISITE_OF'] = len(self.relations)
        self.relations['RELATED_TO'] = len(self.relations)

        self.id2entity = {v: k for k, v in self.entities.items()}
        self.id2relation = {v: k for k, v in self.relations.items()}

        bt_edges = client.run_query("""
            MATCH (p:Problem)-[:BELONGS_TO]->(t:Topic)
            RETURN p.problem_id AS pid, t.name AS tname
        """)
        for r in bt_edges:
            h = self.entities.get(f"problem:{r['pid']}")
            t = self.entities.get(f"topic:{r['tname']}")
            if h is not None and t is not None:
                rel = self.relations['BELONGS_TO']
                self.triples.append((h, rel, t))

        po_edges = client.run_query("""
            MATCH (t1:Topic)-[:PREREQUISITE_OF]->(t2:Topic)
            RETURN t1.name AS source, t2.name AS target
        """)
        for r in po_edges:
            h = self.entities.get(f"topic:{r['source']}")
            t = self.entities.get(f"topic:{r['target']}")
            if h is not None and t is not None:
                rel = self.relations['PREREQUISITE_OF']
                self.triples.append((h, rel, t))

        rt_edges = client.run_query("""
            MATCH (t1:Topic)-[:RELATED_TO]->(t2:Topic)
            RETURN t1.name AS source, t2.name AS target
        """)
        for r in rt_edges:
            h = self.entities.get(f"topic:{r['source']}")
            t = self.entities.get(f"topic:{r['target']}")
            if h is not None and t is not None:
                rel = self.relations['RELATED_TO']
                self.triples.append((h, rel, t))

        print(f"知识图谱数据: {len(self.entities)} 实体, {len(self.relations)} 关系, {len(self.triples)} 三元组")
        return self

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'entities': self.entities,
                'relations': self.relations,
                'id2entity': self.id2entity,
                'id2relation': self.id2relation,
                'triples': self.triples,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.entities = data['entities']
            self.relations = data['relations']
            self.id2entity = data['id2entity']
            self.id2relation = data['id2relation']
            self.triples = data['triples']
        return self


class TransETrainer:
    def __init__(self, save_dir='recommend_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, epochs=200, batch_size=512, lr=0.001, margin=1.0, embed_dim=128):
        kg_path = os.path.join(self.save_dir, 'kg_data.pkl')
        if os.path.exists(kg_path):
            kg = KnowledgeGraphData().load(kg_path)
        else:
            kg = KnowledgeGraphData().build_from_neo4j()
            kg.save(kg_path)

        num_entities = len(kg.entities)
        num_relations = len(kg.relations)
        triples = np.array(kg.triples)

        model = TransE(num_entities, num_relations, embed_dim=embed_dim, margin=margin)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        n = len(triples)
        all_entities = torch.arange(num_entities)

        for epoch in range(epochs):
            model.train()
            np.random.shuffle(triples)
            total_loss = 0
            batches = 0

            for i in range(0, n, batch_size):
                batch_triples = triples[i:i+batch_size]
                pos_h = torch.tensor(batch_triples[:, 0], dtype=torch.long)
                pos_r = torch.tensor(batch_triples[:, 1], dtype=torch.long)
                pos_t = torch.tensor(batch_triples[:, 2], dtype=torch.long)

                neg_t = all_entities[torch.randint(0, num_entities, (len(pos_h),))]
                neg_h = all_entities[torch.randint(0, num_entities, (len(pos_h),))]

                pos_score = model(pos_h, pos_r, pos_t)
                neg_score_h = model(neg_h, pos_r, pos_t)
                neg_score_t = model(pos_h, pos_r, neg_t)

                loss = torch.mean(F.relu(margin + pos_score - neg_score_h)) + \
                       torch.mean(F.relu(margin + pos_score - neg_score_t))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

                with torch.no_grad():
                    model.entity_emb.weight.data.div_(
                        torch.norm(model.entity_emb.weight.data, dim=1, keepdim=True)
                    )

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    sample_h = torch.tensor(triples[:1000, 0], dtype=torch.long)
                    sample_r = torch.tensor(triples[:1000, 1], dtype=torch.long)
                    sample_t = torch.tensor(triples[:1000, 2], dtype=torch.long)
                    pos_scores = model.score_triple(sample_h, sample_r, sample_t)
                    ranks = []
                    for j in range(min(200, len(sample_h))):
                        corrupted_t = torch.randint(0, num_entities, (100,))
                        corr_scores = model.score_triple(
                            sample_h[j].repeat(100),
                            sample_r[j].repeat(100),
                            corrupted_t
                        )
                        rank = (corr_scores > pos_scores[j]).sum().item() + 1
                        ranks.append(rank)
                    mr = np.mean(ranks)
                    mrr = np.mean([1.0 / r for r in ranks])
                    hits10 = sum(1 for r in ranks if r <= 10) / len(ranks)
                print(f"Epoch {epoch+1}, Loss: {total_loss/batches:.4f}, MR: {mr:.1f}, MRR: {mrr:.4f}, Hits@10: {hits10:.4f}")

        torch.save(model.state_dict(), os.path.join(self.save_dir, 'transe.pt'))

        embeddings = model.entity_emb.weight.detach().cpu().numpy()
        embedding_dict = {
            'entities': kg.entities,
            'id2entity': kg.id2entity,
            'embeddings': embeddings,
            'embed_dim': embed_dim,
        }
        with open(os.path.join(self.save_dir, 'kg_embeddings.pkl'), 'wb') as f:
            pickle.dump(embedding_dict, f)

        print("TransE 模型与嵌入已保存")
        return model, embedding_dict

    def discover_missing_links(self, top_k=20):
        model_path = os.path.join(self.save_dir, 'transe.pt')
        kg_path = os.path.join(self.save_dir, 'kg_data.pkl')
        if not os.path.exists(model_path) or not os.path.exists(kg_path):
            print("未找到模型，请先训练")
            return []

        kg = KnowledgeGraphData().load(kg_path)
        num_entities = len(kg.entities)
        num_relations = len(kg.relations)

        model = TransE(num_entities, num_relations)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        prereq_rel = kg.relations.get('PREREQUISITE_OF')
        if prereq_rel is None:
            return []

        existing_pairs = set()
        for h, r, t in kg.triples:
            if r == prereq_rel:
                existing_pairs.add((h, t))

        topic_entities = [eid for eid, idx in kg.entities.items() if eid.startswith('topic:')]
        topic_indices = [kg.entities[eid] for eid in topic_entities]

        candidates = []
        with torch.no_grad():
            for i, h_idx in enumerate(topic_indices):
                for t_idx in topic_indices[i+1:]:
                    if (h_idx, t_idx) in existing_pairs or (t_idx, h_idx) in existing_pairs:
                        continue
                    head = torch.tensor([h_idx], dtype=torch.long)
                    rel = torch.tensor([prereq_rel], dtype=torch.long)
                    tail = torch.tensor([t_idx], dtype=torch.long)
                    score = model.score_triple(head, rel, tail).item()
                    candidates.append((h_idx, t_idx, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        results = []
        for h_idx, t_idx, score in candidates[:top_k]:
            h_name = kg.id2entity[h_idx].replace('topic:', '')
            t_name = kg.id2entity[t_idx].replace('topic:', '')
            results.append({
                'source': h_name,
                'target': t_name,
                'score': float(score),
            })

        return results

    def get_topic_embeddings(self):
        emb_path = os.path.join(self.save_dir, 'kg_embeddings.pkl')
        if not os.path.exists(emb_path):
            return {}

        with open(emb_path, 'rb') as f:
            data = pickle.load(f)

        topic_embs = {}
        for eid, idx in data['entities'].items():
            if eid.startswith('topic:'):
                topic_embs[eid.replace('topic:', '')] = data['embeddings'][idx].tolist()
        return topic_embs

    def get_problem_embeddings(self):
        emb_path = os.path.join(self.save_dir, 'kg_embeddings.pkl')
        if not os.path.exists(emb_path):
            return {}

        with open(emb_path, 'rb') as f:
            data = pickle.load(f)

        prob_embs = {}
        for eid, idx in data['entities'].items():
            if eid.startswith('problem:'):
                prob_embs[eid.replace('problem:', '')] = data['embeddings'][idx].tolist()
        return prob_embs


def load_transe(model_dir='recommend_models'):
    model_path = os.path.join(model_dir, 'transe.pt')
    kg_path = os.path.join(model_dir, 'kg_data.pkl')
    emb_path = os.path.join(model_dir, 'kg_embeddings.pkl')

    if not os.path.exists(model_path) or not os.path.exists(kg_path):
        return None, None, None

    kg = KnowledgeGraphData().load(kg_path)
    model = TransE(len(kg.entities), len(kg.relations))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    embeddings = None
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            embeddings = pickle.load(f)

    return model, kg, embeddings


if __name__ == '__main__':
    trainer = TransETrainer()
    trainer.train(epochs=150)

    print("\n=== 发现缺失的 PREREQUISITE_OF 关系 ===")
    missing = trainer.discover_missing_links(top_k=15)
    for item in missing:
        print(f"  {item['source']} → {item['target']} (score={item['score']:.4f})")
