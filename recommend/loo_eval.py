"""LOO (Leave-One-Out) evaluation for main experiment.
For each user: last accepted problem = test, rest = train.
Runs all baselines + KG rules on LOO protocol.
"""
import os, sys
sys.path.insert(0, "/app")
os.environ["DJANGO_SETTINGS_MODULE"] = "oj.settings"
import django; django.setup()

import numpy as np
from collections import defaultdict
from account.models import User
from problem.models import Problem
from submission.models import Submission, JudgeStatus
from recommend.baselines import Popularity, UserCF, ItemCF
from recommend.evaluate import evaluate_model

# Build LOO split: last item per user = test
users = list(User.objects.values_list("id", flat=True).order_by("id"))
probs = list(Problem.objects.values_list("id", flat=True).order_by("id"))
u2id = {u: i for i, u in enumerate(users)}
p2id = {p: i for i, p in enumerate(probs)}
id2u = {v: k for k, v in u2id.items()}

acc = Submission.objects.filter(result=JudgeStatus.ACCEPTED).order_by("create_time")
ui = {}
for sub in acc:
    u = sub.user_id; p = sub.problem_id
    if u not in u2id or p not in p2id: continue
    ui.setdefault(u2id[u], []).append(p2id[p])

X_tr, X_te = [], []
for ui_, items in ui.items():
    if len(items) < 2: continue
    # LOO: last item = test
    X_tr += [(ui_, i) for i in items[:-1]]
    X_te.append((ui_, items[-1]))

print("LOO split: {} train pairs, {} test pairs, {} test users".format(
    len(X_tr), len(X_te), len(set(u for u,_ in X_te))))

# Negative sampling for train (BPR-MF needs neg)
neg_pairs = []
for u_inner in range(len(users)):
    user_pos = set(p for (ui, p) in X_tr if ui == u_inner)
    available = [i for i in range(len(probs)) if i not in user_pos]
    num_neg = min(len(user_pos), 4) if user_pos else 4
    if len(available) >= num_neg:
        for p in np.random.choice(available, size=num_neg, replace=False):
            neg_pairs.append((u_inner, int(p)))
X_tr_all = X_tr + neg_pairs

class Wrap:
    def __init__(self, fn): self.fn = fn
    def recommend(self, u, k=10): return self.fn(u, k=k)

# Pop
pop = Popularity(len(users), len(probs))
pop.fit(X_tr)
m = evaluate_model(Wrap(pop.recommend), X_te, len(probs), ks=[5,10,20])
print("Popularity:  R@5={:.4f} R@10={:.4f} R@20={:.4f} N@10={:.4f} H@10={:.4f} MRR={:.4f}".format(
    m['Recall@5'], m['Recall@10'], m['Recall@20'], m['NDCG@10'], m['Hit@10'], m['MRR']))

# UserCF
ucf = UserCF(len(users), len(probs), K=50)
ucf.fit(X_tr)
m = evaluate_model(Wrap(ucf.recommend), X_te, len(probs), ks=[5,10,20])
print("UserCF:      R@5={:.4f} R@10={:.4f} R@20={:.4f} N@10={:.4f} H@10={:.4f} MRR={:.4f}".format(
    m['Recall@5'], m['Recall@10'], m['Recall@20'], m['NDCG@10'], m['Hit@10'], m['MRR']))

# ItemCF
icf = ItemCF(len(users), len(probs), K=50)
icf.fit(X_tr)
m = evaluate_model(Wrap(icf.recommend), X_te, len(probs), ks=[5,10,20])
print("ItemCF:      R@5={:.4f} R@10={:.4f} R@20={:.4f} N@10={:.4f} H@10={:.4f} MRR={:.4f}".format(
    m['Recall@5'], m['Recall@10'], m['Recall@20'], m['NDCG@10'], m['Hit@10'], m['MRR']))

# BPR-MF
from recommend.baselines import BPRMF
r10s = []
for seed in range(3):
    np.random.seed(seed)
    bpr = BPRMF(len(users), len(probs), embed_dim=64, lr=0.01, epochs=50)
    bpr.fit(X_tr_all)
    m = evaluate_model(Wrap(bpr.recommend), X_te, len(probs), ks=[5,10,20])
    r10s.append(m['Recall@10'])
print("BPR-MF:      R@5={:.4f} R@10={:.4f}(std={:.4f}) R@20={:.4f} N@10={:.4f} H@10={:.4f} MRR={:.4f}".format(
    m['Recall@5'], np.mean(r10s), np.std(r10s), m['Recall@20'], m['NDCG@10'], m['Hit@10'], m['MRR']))

# KG-Rules (requires Neo4j)
from agents.recommend_agent import _get_graph_recommendations, _get_hot_recommendations
ukg = []
for ui_, uid in [(u, id2u.get(u)) for u in set(u_ for u_,_ in X_te)]:
    if uid is None: continue
    u = User.objects.filter(id=uid).first()
    if not u: continue
    recs = _get_graph_recommendations(u.username, limit=50)
    ranked = []
    for r in recs:
        if r["id"] in p2id: ranked.append((p2id[r["id"]], r["score"]/100.0))
    if len(ranked) < 10:
        hot = _get_hot_recommendations(u, limit=20)
        for r in hot:
            pid = r["id"]
            if pid in p2id and p2id[pid] not in {i for i,_ in ranked}:
                ranked.append((p2id[pid], 0.05))
            if len(ranked) >= 10: break
    ukg.append((ui_, ranked[:10]))

rk5, rk10, rk20, n10, h10, mrr = [], [], [], [], [], []
for u_i, recs in ukg:
    pos = {i for uu, i in X_te if uu == u_i}
    ranked = [i for i,_ in recs]
    hits = np.array([1.0 if i in pos else 0.0 for i in ranked])
    rk5.append(hits[:5].sum()/max(len(pos),1))
    rk10.append(hits[:10].sum()/max(len(pos),1))
    rk20.append(hits[:20].sum()/max(len(pos),1))
    h10.append(1.0 if hits[:10].sum()>0 else 0.0)
    dcg = sum(1.0/np.log2(j+2) for j,h in enumerate(hits[:10]) if h>0)
    idcg = sum(1.0/np.log2(j+2) for j in range(min(10,len(pos))))
    n10.append(dcg/idcg if idcg>0 else 0.0)
    for rank, item in enumerate(ranked[:20], 1):
        if item in pos: mrr.append(1.0/rank); break
    else: mrr.append(0.0)

print("KG-Rules:    R@5={:.4f} R@10={:.4f} R@20={:.4f} N@10={:.4f} H@10={:.4f} MRR={:.4f}".format(
    np.mean(rk5), np.mean(rk10), np.mean(rk20), np.mean(n10), np.mean(h10), np.mean(mrr)))
