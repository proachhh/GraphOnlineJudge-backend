"""Quick evaluation of Codeforces full dataset"""
import sys, pickle, numpy as np, logging, time
from collections import defaultdict
sys.path.insert(0, '/home/proach/OJ/recommend')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

from run_cf_experiments import recall_at_k, ndcg_at_k, hit_at_k, mrr, KGRecommender

with open('/home/proach/publicData/processed/codeforces_uall_min5.pkl', 'rb') as f:
    d = pickle.load(f)
logger.info(f'Loaded: {d["n_users"]} users, {d["n_items"]} items, {d["n_train"]} train, {d["n_test"]} test')

X_train, X_test = d['X_train'], d['X_test']
ks = [5, 10, 20]

# Group test by user
ut = defaultdict(set)
for u, p in X_test:
    ut[u].add(p)

# Popularity
counts = defaultdict(int)
for u, p in X_train:
    counts[p] += 1
pop = sorted(counts.items(), key=lambda x: -x[1])
pop_recs = [(p, 1.0) for p, _ in pop]

m = {f'R@{k}': [] for k in ks}
m.update({f'N@{k}': [] for k in ks})
m.update({f'H@{k}': [] for k in ks})
m['MRR'] = []
t0 = time.time()
for u in sorted(ut.keys()):
    true = ut[u]
    recs = pop_recs[:max(ks)]
    for k in ks:
        m[f'R@{k}'].append(recall_at_k(recs, true, k))
        m[f'N@{k}'].append(ndcg_at_k(recs, true, k))
        m[f'H@{k}'].append(hit_at_k(recs, true, k))
    m['MRR'].append(mrr(recs, true))
print(f'Pop:    R@5={np.mean(m["R@5"]):.4f} R@10={np.mean(m["R@10"]):.4f} R@20={np.mean(m["R@20"]):.4f} N@10={np.mean(m["N@10"]):.4f} H@10={np.mean(m["H@10"]):.4f} MRR={np.mean(m["MRR"]):.4f} ({time.time()-t0:.1f}s)')

# KG
t0 = time.time()
kg = KGRecommender(d)
kg.fit(X_train)
m = {f'R@{k}': [] for k in ks}
m.update({f'N@{k}': [] for k in ks})
m.update({f'H@{k}': [] for k in ks})
m['MRR'] = []
for u in sorted(ut.keys()):
    true = ut[u]
    recs = kg.recommend(u, max(ks))
    for k in ks:
        m[f'R@{k}'].append(recall_at_k(recs, true, k))
        m[f'N@{k}'].append(ndcg_at_k(recs, true, k))
        m[f'H@{k}'].append(hit_at_k(recs, true, k))
    m['MRR'].append(mrr(recs, true))
print(f'KG:     R@5={np.mean(m["R@5"]):.4f} R@10={np.mean(m["R@10"]):.4f} R@20={np.mean(m["R@20"]):.4f} N@10={np.mean(m["N@10"]):.4f} H@10={np.mean(m["H@10"]):.4f} MRR={np.mean(m["MRR"]):.4f} ({time.time()-t0:.1f}s)')

# Topic coherence
print("\n=== Topic Coherence ===")
tags = d['all_tags']
pids = d['id2prob']
prob_tags = d['prob_tags']

# Build user known topics
user_known = {}
for u, p in X_train:
    pid = pids.get(p, '')
    ts = prob_tags.get(pid, [])
    if u not in user_known:
        user_known[u] = set()
    user_known[u].update(ts)

# Pop coherence
pc = []
for u in sorted(ut.keys()):
    if u not in user_known or not user_known[u]:
        continue
    known = user_known[u]
    for p in pop_recs[:10]:
        pid = pids.get(p[0], '')
        rt = prob_tags.get(pid, [])
        if rt:
            pc.append(1.0 if (set(rt) & known) else 0.0)
print(f'Pop coherence: {np.mean(pc)*100:.1f}% (n={len(pc)})')

# KG coherence
kc = []
for u in sorted(ut.keys()):
    if u not in user_known or not user_known[u]:
        continue
    known = user_known[u]
    recs = kg.recommend(u, 10)
    for p, _ in recs[:10]:
        pid = pids.get(p, '')
        rt = prob_tags.get(pid, [])
        if rt:
            kc.append(1.0 if (set(rt) & known) else 0.0)
print(f'KG coherence:  {np.mean(kc)*100:.1f}% (n={len(kc)})')
