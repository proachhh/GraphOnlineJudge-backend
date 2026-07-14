"""Cold-start evaluation: compare KG vs baselines on users with <=3 submissions."""
import os, sys
sys.path.insert(0, "/app")
os.environ["DJANGO_SETTINGS_MODULE"] = "oj.settings"
import django; django.setup()

import numpy as np
from account.models import User
from problem.models import Problem
from submission.models import Submission, JudgeStatus
from recommend.baselines import Popularity, UserCF, ItemCF
from recommend.evaluate import evaluate_model
from agents.recommend_agent import _get_graph_recommendations, _get_hot_recommendations

# Temporal split
users = list(User.objects.values_list("id", flat=True).order_by("id"))
probs = list(Problem.objects.values_list("id", flat=True).order_by("id"))
u2id = {u: i for i, u in enumerate(users)}
p2id = {p: i for i, p in enumerate(probs)}
id2u = {v: k for k, v in u2id.items()}

# Cold users: <=3 accepted
uc = {}
for s in Submission.objects.filter(result=JudgeStatus.ACCEPTED).values_list("user_id", flat=True):
    uc[s] = uc.get(s, 0) + 1
cold_set = set(uid for uid, cnt in uc.items() if cnt <= 3)

# Build temporal split
acc = Submission.objects.filter(result=JudgeStatus.ACCEPTED).order_by("create_time")
ui = {}
for sub in acc:
    u = sub.user_id
    p = sub.problem_id
    if u not in u2id or p not in p2id:
        continue
    ui.setdefault(u2id[u], []).append(p2id[p])

X_tr, X_te = [], []
for ui_, items in ui.items():
    if len(items) < 2:
        continue
    sp = max(1, int(len(items) * 0.8))
    X_tr += [(ui_, i) for i in items[:sp]]
    X_te += [(ui_, i) for i in items[sp:]]

# Cold-start test subset
cold_test = [(u, i) for u, i in X_te if id2u.get(u) in cold_set]
print("Cold-start test: {} pairs, {} users".format(len(cold_test), len(set(u for u, _ in cold_test))))

# ---- Baselines ----
# Pop
pop = Popularity(len(users), len(probs))
pop.fit(X_tr)
class Wrap:
    def __init__(self, fn, k=10):
        self.fn = fn
        self.k = k
    def recommend(self, u, k=10):
        return self.fn(u, k=k)

m = evaluate_model(Wrap(pop.recommend), cold_test, len(probs), ks=[5, 10])
print("Popularity:  R@5={:.4f} R@10={:.4f} N@10={:.4f}".format(m['Recall@5'], m['Recall@10'], m['NDCG@10']))

# UserCF
ucf = UserCF(len(users), len(probs), K=50)
ucf.fit(X_tr)
m = evaluate_model(Wrap(ucf.recommend), cold_test, len(probs), ks=[5, 10])
print("UserCF:      R@5={:.4f} R@10={:.4f} N@10={:.4f}".format(m['Recall@5'], m['Recall@10'], m['NDCG@10']))

# ItemCF
icf = ItemCF(len(users), len(probs), K=50)
icf.fit(X_tr)
m = evaluate_model(Wrap(icf.recommend), cold_test, len(probs), ks=[5, 10])
print("ItemCF:      R@5={:.4f} R@10={:.4f} N@10={:.4f}".format(m['Recall@5'], m['Recall@10'], m['NDCG@10']))

# ---- KG-Rules (ours) ----
ukg_r = []
for ui_, items in ui.items():
    uid = id2u[ui_]
    if uid not in cold_set:
        continue
    u = User.objects.filter(id=uid).first()
    if not u:
        continue
    recs = _get_graph_recommendations(u.username, limit=50)
    ranked = []
    for r in recs:
        if r["id"] in p2id:
            ranked.append((p2id[r["id"]], r["score"] / 100.0))
    # Hot fallback
    if len(ranked) < 10:
        hot = _get_hot_recommendations(u, limit=20)
        for r in hot:
            pid = r["id"]
            if pid in p2id and p2id[pid] not in {i for i, _ in ranked}:
                ranked.append((p2id[pid], 0.05))
            if len(ranked) >= 10:
                break
    ukg_r.append((ui_, ranked[:10]))

# Manual evaluation
rk5, rk10, n10 = [], [], []
for u_i, recs in ukg_r:
    pos = set(i for uu, i in cold_test if uu == u_i)
    ranked = [i for i, _ in recs]
    hits = np.array([1.0 if i in pos else 0.0 for i in ranked])
    rk5.append(min(1.0, hits[:5].sum() / max(len(pos), 1)))
    rk10.append(min(1.0, hits[:10].sum() / max(len(pos), 1)))
    dcg = sum(1.0 / np.log2(j + 2) for j, h in enumerate(hits[:10]) if h > 0)
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(10, len(pos))))
    n10.append(dcg / idcg if idcg > 0 else 0.0)

print("KG-Rules:    R@5={:.4f} R@10={:.4f} N@10={:.4f}".format(np.mean(rk5), np.mean(rk10), np.mean(n10)))
