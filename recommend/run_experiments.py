"""
Main Experiment Runner for Paper Evaluation.

Runs:
  1. Baseline models (Popularity, UserCF, ItemCF, BPR-MF)
  2. Our method: Knowledge Graph rules (graph recall)
  3. Our method: GNN recall
  4. Our method: Sequence recall  
  5. Our method: Fusion (graph + GNN + sequence)
  6. Our method: Fusion + DeepFM rerank (full pipeline)
  7. Ablation: w/o GNN, w/o Sequence, w/o DeepFM

Outputs:
  - Printed comparison table
  - LaTeX table for paper
  - JSON results for plots
"""

import os
import sys

# Ensure the project root is in sys.path (needed inside Docker)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django
django.setup()

from recommend.baselines import (
    Popularity, UserCF, ItemCF, BPRMF, load_data, train_baselines
)
from recommend.evaluate import (
    evaluate_model, evaluate_your_model, print_results, format_results_table
)
from account.models import User
from submission.models import Submission, JudgeStatus
from problem.models import Problem


def build_temporal_split(test_ratio=0.2, min_interactions=2):
    """
    Build train/test split using temporal ordering (standard recsys evaluation).
    For each user: earliest (1-test_ratio) submissions → train, latest → test.
    Only includes users with >= min_interactions accepted problems.
    """
    logger.info("Building temporal split from submissions...")
    
    # Get all users and problems (same as data_export)
    users = list(User.objects.values_list('id', flat=True).order_by('id'))
    problems = list(Problem.objects.values_list('id', flat=True).order_by('id'))
    user2id = {uid: i for i, uid in enumerate(users)}
    prob2id = {pid: i for i, pid in enumerate(problems)}
    
    # Get accepted submissions ordered by time
    accepted = Submission.objects.filter(
        result=JudgeStatus.ACCEPTED
    ).select_related('problem').order_by('create_time')
    
    # Group by user, ordered by time
    user_items = {}
    for sub in accepted:
        uid = sub.user_id
        pid = sub.problem_id
        if uid not in user2id or pid not in prob2id:
            continue
        u_inner = user2id[uid]
        p_inner = prob2id[pid]
        if u_inner not in user_items:
            user_items[u_inner] = []
        if p_inner not in user_items[u_inner]:
            user_items[u_inner].append(p_inner)
    
    # Temporal split per user
    X_train, X_test = [], []
    qualified_users = 0
    
    for u_inner, items in user_items.items():
        if len(items) < min_interactions:
            continue
        qualified_users += 1
        split_idx = max(1, int(len(items) * (1 - test_ratio)))
        for p_inner in items[:split_idx]:
            X_train.append((u_inner, p_inner))
        for p_inner in items[split_idx:]:
            X_test.append((u_inner, p_inner))
    
    # Negative sampling for train (not for test - we use ranking evaluation)
    pos_count = len(X_train)
    neg_pairs = []
    total_items = len(problems)
    for u_inner in range(len(users)):
        user_pos = set(p for (ui, p) in X_train if ui == u_inner)
        available = [i for i in range(total_items) if i not in user_pos]
        num_neg = min(len(user_pos), 4) if user_pos else 4
        if len(available) >= num_neg:
            neg_samples = np.random.choice(available, size=num_neg, replace=False)
            for p in neg_samples:
                neg_pairs.append((u_inner, int(p)))
    
    X_train_all = X_train + neg_pairs
    y_train = [1] * len(X_train) + [0] * len(neg_pairs)
    
    logger.info(f"Temporal split: {qualified_users}/{len(users)} users qualified, "
                f"{len(X_train)} pos train, {len(X_test)} pos test, {len(neg_pairs)} neg train")
    
    return {
        'X_train': X_train_all,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': [1] * len(X_test),
        'num_users': len(users),
        'num_items': len(problems),
        'user2id': user2id,
        'prob2id': prob2id,
    }


def build_user_item_index(data):
    """Build user_id -> inner_idx and item_id -> problem_id mappings."""
    user2id = data['user2id']  # user_id -> inner_idx
    prob2id = data['prob2id']  # problem_id -> inner_idx
    
    id2user = {v: k for k, v in user2id.items()}  # inner_idx -> user_id
    id2prob = {v: k for k, v in prob2id.items()}  # inner_idx -> problem_id
    
    return id2user, id2prob


def run_baseline_experiments(data, models, ks=[5, 10, 20]):
    """Evaluate all baseline models."""
    results = {}
    X_test = data['X_test']
    num_items = data['num_items']
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_test, num_items, ks=ks)
        results[name] = metrics
        logger.info(f"  {name}: Recall@10={metrics.get('Recall@10', 0):.4f}, "
                     f"NDCG@10={metrics.get('NDCG@10', 0):.4f}")
    
    return results


def build_graph_hybrid_recommender(user_id, data, id2user):
    """
    Build a recommender that wraps the knowledge graph rule-based + CF methods
    from recommend_agent.py, but returns inner indices.
    """
    from agents.recommend_agent import (
        _get_graph_recommendations, _get_cf_recommendations, _get_hot_recommendations
    )
    from account.models import User
    
    prob2id = data['prob2id']
    user = User.objects.filter(id=user_id).first()
    if not user:
        return []
    
    username = user.username
    seen = set()
    recs = []
    
    # Graph rules
    graph_recs = _get_graph_recommendations(username, limit=50)
    for r in graph_recs:
        pid = r['id']
        if pid not in seen and pid in prob2id:
            seen.add(pid)
            recs.append((prob2id[pid], r['score'] / 100.0))
    
    # CF
    cf_recs = _get_cf_recommendations(username, limit=30)
    for r in cf_recs:
        pid = r['id']
        if pid not in seen and pid in prob2id:
            seen.add(pid)
            recs.append((prob2id[pid], r['score'] / 100.0))
    
    return recs


def build_gnn_recommender(data):
    """GNN recall recommender."""
    from recommend.multi_modal_fusion import MultiModalFusionEngine
    
    prob2id = data['prob2id']
    engine = MultiModalFusionEngine('/data/recommend_models')
    
    if engine.gnn_embeddings is None:
        return None
    
    def recommend(user_id, k=10):
        try:
            candidates = engine.gnn_recall(user_id, top_k=k)
        except Exception:
            return []
        
        recs = []
        seen = set()
        for c in candidates:
            pid = int(c.problem_id) if c.problem_id else 0
            if pid not in seen and pid in prob2id:
                seen.add(pid)
                recs.append((prob2id[pid], c.score))
        return recs
    
    return recommend


def build_sequence_recommender(data):
    """Sequence recall recommender."""
    from recommend.multi_modal_fusion import MultiModalFusionEngine
    
    prob2id = data['prob2id']
    engine = MultiModalFusionEngine('/data/recommend_models')
    
    if engine.sequence_model is None:
        return None
    
    def recommend(user_id, k=10):
        try:
            candidates = engine.sequence_recall(user_id, top_k=k)
        except Exception:
            return []
        
        recs = []
        seen = set()
        for c in candidates:
            pid = int(c.problem_id) if c.problem_id else 0
            if pid not in seen and pid in prob2id:
                seen.add(pid)
                recs.append((prob2id[pid], c.score))
        return recs
    
    return recommend


def build_fusion_recommender(data, use_gnn=True, use_sequence=True, use_rerank=False,
                            popularity_model=None):
    """
    Build fusion recommender with optional components for ablation.
    Includes popularity fallback for cold users (same as production system).
    """
    from recommend.multi_modal_fusion import MultiModalFusionEngine
    
    prob2id = data['prob2id']
    engine = MultiModalFusionEngine('/data/recommend_models')
    # Pre-compute popular items for fallback
    if popularity_model is not None and popularity_model.sorted_items is not None:
        popular_items = [int(i) for i in popularity_model.sorted_items[:100]]
    else:
        popular_items = list(range(data['num_items']))
    
    def recommend(user_id, k=10):
        all_candidates = []
        seen = set()
        
        # Graph rules + CF (always included as base)
        graph_recs = build_graph_hybrid_recommender(user_id, data, None)
        for inner_idx, score in graph_recs:
            if inner_idx not in seen:
                seen.add(inner_idx)
                all_candidates.append({'inner_idx': inner_idx, 'score': score})
        
        if use_gnn and engine.gnn_embeddings is not None:
            try:
                gnn_cands = engine.gnn_recall(user_id, top_k=k)
                for c in gnn_cands:
                    pid = int(c.problem_id) if c.problem_id else 0
                    inner = prob2id.get(pid)
                    if inner is not None and inner not in seen:
                        seen.add(inner)
                        all_candidates.append({'inner_idx': inner, 'score': c.score})
            except Exception:
                pass
        
        if use_sequence and engine.sequence_model is not None:
            try:
                seq_cands = engine.sequence_recall(user_id, top_k=k)
                for c in seq_cands:
                    pid = int(c.problem_id) if c.problem_id else 0
                    inner = prob2id.get(pid)
                    if inner is not None and inner not in seen:
                        seen.add(inner)
                        all_candidates.append({'inner_idx': inner, 'score': c.score})
            except Exception:
                pass
        
        # Always include popular items as fallback (low priority, only shown if KG/GNN/Seq miss)
        for item in popular_items[:k*2]:
            if item not in seen:
                seen.add(item)
                all_candidates.append({'inner_idx': item, 'score': 0.05})
        
        # Interleave: k//3 graph items + fill with popular for coverage
        graph_candidates = [c for c in all_candidates if c['score'] > 0.1]
        pop_candidates = [c for c in all_candidates if c['score'] <= 0.1]
        graph_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        all_candidates = graph_candidates[:max(1, k // 3)] + pop_candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if use_rerank and engine.ranking_model is not None:
            try:
                from recommend.deepfm_ranking import deepfm_rank
                candidate_pids = []
                for c in all_candidates[:50]:
                    inner = c['inner_idx']
                    for pid, iid in prob2id.items():
                        if iid == inner:
                            candidate_pids.append(pid)
                            break
                ranked = deepfm_rank(user_id, candidate_pids, engine.ranking_model, engine.ranking_data)
                score_map = dict(ranked)
                for c in all_candidates:
                    for pid, iid in prob2id.items():
                        if iid == c['inner_idx'] and pid in score_map:
                            c['score'] = score_map[pid]
                            break
                all_candidates.sort(key=lambda x: x['score'], reverse=True)
            except Exception:
                pass
        
        return [(c['inner_idx'], c['score']) for c in all_candidates[:k]]
    
    return recommend


def run_our_method_experiments(data, id2user, popularity_model=None, ks=[5, 10, 20]):
    """Evaluate all our method variants."""
    results = {}
    
    # 1. Graph rules only (KG-based)
    logger.info("Evaluating KG Graph Rules...")
    
    class KGRecommender:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return build_graph_hybrid_recommender(user_id, data, id2user)[:k]
    
    kg_model = KGRecommender()
    metrics = evaluate_model(kg_model, data['X_test'], data['num_items'], ks=ks)
    results['KG-Rules'] = metrics
    logger.info(f"  KG-Rules: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 2. GNN only
    logger.info("Evaluating GNN Recall...")
    gnn_fn = build_gnn_recommender(data)
    if gnn_fn:
        class GNNRecommender:
            def recommend(self, user_idx, k=10):
                user_id = id2user.get(user_idx)
                if user_id is None:
                    return []
                return gnn_fn(user_id, k=k)
        
        gnn_model = GNNRecommender()
        metrics = evaluate_model(gnn_model, data['X_test'], data['num_items'], ks=ks)
        results['GNN'] = metrics
        logger.info(f"  GNN: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 3. Sequence only
    logger.info("Evaluating Sequence Recall...")
    seq_fn = build_sequence_recommender(data)
    if seq_fn:
        class SeqRecommender:
            def recommend(self, user_idx, k=10):
                user_id = id2user.get(user_idx)
                if user_id is None:
                    return []
                return seq_fn(user_id, k=k)
        
        seq_model = SeqRecommender()
        metrics = evaluate_model(seq_model, data['X_test'], data['num_items'], ks=ks)
        results['Transformer-Seq'] = metrics
        logger.info(f"  Transformer-Seq: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 4. Full Fusion (KG + GNN + Seq)
    logger.info("Evaluating Full Fusion...")
    fusion_fn = build_fusion_recommender(data, use_gnn=True, use_sequence=True, use_rerank=False, popularity_model=popularity_model)
    
    class FusionRecommender:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return fusion_fn(user_id, k=k)
    
    fusion_model = FusionRecommender()
    metrics = evaluate_model(fusion_model, data['X_test'], data['num_items'], ks=ks)
    results['Fusion (ours)'] = metrics
    logger.info(f"  Fusion (ours): Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 5. Full pipeline with DeepFM rerank
    logger.info("Evaluating Fusion + DeepFM...")
    full_fn = build_fusion_recommender(data, use_gnn=True, use_sequence=True, use_rerank=True, popularity_model=popularity_model)
    
    class FullRecommender:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return full_fn(user_id, k=k)
    
    full_model = FullRecommender()
    metrics = evaluate_model(full_model, data['X_test'], data['num_items'], ks=ks)
    results['Fusion+DeepFM (ours)'] = metrics
    logger.info(f"  Fusion+DeepFM (ours): Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 6. Ablation: w/o GNN
    logger.info("Evaluating Ablation: w/o GNN...")
    ablation_no_gnn_fn = build_fusion_recommender(data, use_gnn=False, use_sequence=True, use_rerank=False, popularity_model=popularity_model)
    
    class AblationNoGNN:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return ablation_no_gnn_fn(user_id, k=k)
    
    no_gnn_model = AblationNoGNN()
    metrics = evaluate_model(no_gnn_model, data['X_test'], data['num_items'], ks=ks)
    results['Ablation w/o GNN'] = metrics
    logger.info(f"  w/o GNN: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 7. Ablation: w/o Sequence
    logger.info("Evaluating Ablation: w/o Sequence...")
    ablation_no_seq_fn = build_fusion_recommender(data, use_gnn=True, use_sequence=False, use_rerank=False, popularity_model=popularity_model)
    
    class AblationNoSeq:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return ablation_no_seq_fn(user_id, k=k)
    
    no_seq_model = AblationNoSeq()
    metrics = evaluate_model(no_seq_model, data['X_test'], data['num_items'], ks=ks)
    results['Ablation w/o Seq'] = metrics
    logger.info(f"  w/o Seq: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    # 8. Ablation: w/o DeepFM
    logger.info("Evaluating Ablation: w/o DeepFM...")
    ablation_no_dfm_fn = build_fusion_recommender(data, use_gnn=True, use_sequence=True, use_rerank=False, popularity_model=popularity_model)
    
    class AblationNoDFM:
        def recommend(self, user_idx, k=10):
            user_id = id2user.get(user_idx)
            if user_id is None:
                return []
            return ablation_no_dfm_fn(user_id, k=k)
    
    no_dfm_model = AblationNoDFM()
    metrics = evaluate_model(no_dfm_model, data['X_test'], data['num_items'], ks=ks)
    results['Ablation w/o DeepFM'] = metrics
    logger.info(f"  w/o DeepFM: Recall@10={metrics.get('Recall@10', 0):.4f}")
    
    return results


def save_results(all_results, output_dir='/data'):
    """Save results to JSON and LaTeX."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, 'experiment_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"Results saved to {json_path}")
    
    # LaTeX table
    latex = format_results_table(all_results)
    latex_path = os.path.join(output_dir, 'results_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    logger.info(f"LaTeX table saved to {latex_path}")


def main():
    logger.info("="*60)
    logger.info("Paper Experiment Runner")
    logger.info("="*60)
    
    # Load data with TEMPORAL SPLIT (standard recsys evaluation)
    logger.info("Loading data...")
    data = build_temporal_split(test_ratio=0.2, min_interactions=2)
    id2user, id2prob = build_user_item_index(data)
    logger.info(f"Data loaded: {data['num_users']} users, {data['num_items']} items")
    
    # Run baselines
    logger.info("\n" + "="*60)
    logger.info("1. Training and evaluating baseline models")
    logger.info("="*60)
    baseline_models = train_baselines(data)
    baseline_results = run_baseline_experiments(data, baseline_models)
    
    # Run our method
    logger.info("\n" + "="*60)
    logger.info("2. Evaluating our proposed method")
    logger.info("="*60)
    our_results = run_our_method_experiments(data, id2user, popularity_model=baseline_models.get('Popularity'))
    
    # Combine
    all_results = {**baseline_results, **our_results}
    
    # Print
    logger.info("\n" + "="*60)
    logger.info("3. Final Results Summary")
    logger.info("="*60)
    print_results(all_results)
    
    # Save
    save_results(all_results)
    
    logger.info("\nExperiment completed!")
    return all_results


if __name__ == '__main__':
    main()
