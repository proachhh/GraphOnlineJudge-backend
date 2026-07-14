"""
Novelty & Diversity Analysis for Paper
- What % of KG recommendations are from new topics?
- What % of ItemCF recommendations are from new topics?
- Coverage and intra-list diversity comparison

Run on server:
  docker exec -w /app onlinejudgedeploy-oj-backend-1 python3 recommend/novelty_analysis.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django; django.setup()

import numpy as np
from collections import defaultdict
from account.models import User
from problem.models import Problem, ProblemTag
from submission.models import Submission, JudgeStatus
from agents.recommend_agent import _get_graph_recommendations, _get_cf_recommendations, _get_hot_recommendations

# ============================================================
# 1. Build per-user topic history
# ============================================================
print("Building user-topic history...")
user_topic_history = {}  # user_id -> set of topic names
user_username = {}

for user in User.objects.all():
    username = user.username
    user_username[user.id] = username
    topics = set(
        ProblemTag.objects.filter(
            problem__submission__user_id=user.id,
            problem__submission__result=JudgeStatus.ACCEPTED
        ).values_list('name', flat=True).distinct()
    )
    user_topic_history[user.id] = topics

# ============================================================
# 2. Get problem -> topics mapping
# ============================================================
print("Building problem-topic mapping...")
problem_topics = {}
for p in Problem.objects.all():
    problem_topics[p.id] = set(p.tags.values_list('name', flat=True))

# ============================================================
# 3. Compute novelty: % of recommended problems from NEW topics
# ============================================================
def compute_novelty(recommendations, user_id):
    """Fraction of recommended problems whose topics the user hasn't seen."""
    if not recommendations:
        return None
    known = user_topic_history.get(user_id, set())
    novel_count = 0
    total = 0
    for rec in recommendations:
        pid = rec['id'] if isinstance(rec, dict) else rec
        rec_topics = problem_topics.get(pid, set())
        if rec_topics:
            total += 1
            if not (rec_topics & known):  # no overlap = novel topic
                novel_count += 1
    return novel_count / total if total > 0 else 0.0

def compute_itemcf_novelty(username, user_id, limit=30):
    """Compute novelty for ItemCF (simulated via _get_cf_recommendations)."""
    recs = _get_cf_recommendations(username, limit=limit)
    return compute_novelty(recs, user_id)

def compute_kg_novelty(username, user_id, limit=50):
    """Compute novelty for KG rules channel."""
    recs = _get_graph_recommendations(username, limit=limit)
    return compute_novelty(recs, user_id)

# ============================================================
# 4. Compute per-user novelty
# ============================================================
print("\nComputing novelty per user...")
kg_novelties = []
cf_novelties = []
count = 0

for user_id, username in user_username.items():
    if len(user_topic_history.get(user_id, set())) < 1:
        continue  # skip users with no topic history
    
    kg_nov = compute_kg_novelty(username, user_id)
    cf_nov = compute_itemcf_novelty(username, user_id)
    
    if kg_nov is not None:
        kg_novelties.append(kg_nov)
    if cf_nov is not None:
        cf_novelties.append(cf_nov)
    
    count += 1
    if count % 10 == 0:
        print(f"  Processed {count} users...")

# ============================================================
# 5. Coverage: % of total problems ever recommended
# ============================================================
print("\nComputing coverage...")
all_problems = set(Problem.objects.values_list('id', flat=True))
kg_covered = set()
cf_covered = set()

for user_id, username in user_username.items():
    kg_recs = _get_graph_recommendations(username, limit=50)
    cf_recs = _get_cf_recommendations(username, limit=30)
    
    for r in kg_recs:
        kg_covered.add(r['id'])
    for r in cf_recs:
        cf_covered.add(r['id'])

kg_coverage = len(kg_covered) / len(all_problems) if all_problems else 0
cf_coverage = len(cf_covered) / len(all_problems) if all_problems else 0

# ============================================================
# 6. Print results
# ============================================================
print("\n" + "="*60)
print("RESULTS FOR PAPER")
print("="*60)

print(f"\nNovelty (% recommendations from unseen topics):")
print(f"  KG-Rules: {np.mean(kg_novelties)*100:.1f}% (avg over {len(kg_novelties)} users)")
print(f"  ItemCF:   {np.mean(cf_novelties)*100:.1f}% (avg over {len(cf_novelties)} users)")

print(f"\nCoverage (% of total problem catalog):")
print(f"  KG-Rules: {kg_coverage*100:.1f}%")
print(f"  ItemCF:   {cf_coverage*100:.1f}%")

print(f"\nDiversity (avg unique topics per top-10 recommendations):")
# Quick diversity: avg # unique topics in top-10
kg_topic_diversity = []
cf_topic_diversity = []
for user_id, username in list(user_username.items())[:20]:  # sample 20 users
    kg_recs = _get_graph_recommendations(username, limit=10)
    cf_recs = _get_cf_recommendations(username, limit=10)
    
    kg_topics = set()
    for r in kg_recs:
        kg_topics |= problem_topics.get(r['id'], set())
    cf_topics = set()
    for r in cf_recs:
        cf_topics |= problem_topics.get(r['id'], set())
    
    kg_topic_diversity.append(len(kg_topics))
    cf_topic_diversity.append(len(cf_topics))

print(f"  KG-Rules: {np.mean(kg_topic_diversity):.1f} unique topics")
print(f"  ItemCF:   {np.mean(cf_topic_diversity):.1f} unique topics")
