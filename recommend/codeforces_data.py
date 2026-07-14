"""
Codeforces Dataset Loader for Recommendation Experiments.

Loads from raw CSV files, filters, builds interaction matrix, knowledge graph,
and provides train/test splits for recommendation evaluation.

Raw data:
  /home/proach/data/codeforces_problems.csv       — 11,148 problems with tags
  /home/proach/data/usersCodeforcesSubmissionsEnd2024.csv — 17.6M submissions

Output (cached to /home/proach/data/processed/):
  codeforces_train.pkl, codeforces_test.pkl, codeforces_kg.pkl
"""

import os
import pickle
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/data/codeforces'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PROBLEMS_CSV = os.path.join(DATA_DIR, 'problems.csv')
SUBMISSIONS_CSV = os.path.join(DATA_DIR, 'submissions.csv')


def ensure_processed_dir():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# STEP 1: Load & filter submissions
# ──────────────────────────────────────────────────────────────────────

def load_and_filter(min_user_interactions: int = 5, max_users: int = None) -> dict:
    """
    Load raw CSVs, filter to AC+Verdict=OK + matching problem IDs,
    build user→item interaction dict.

    Args:
        min_user_interactions: minimum distinct solved problems per user
        max_users: cap number of users (None = all)

    Returns dict with:
        user_items[user_str] = [(pid_str, timestamp, rating, tags), ...]
        prob_tags[pid_str] = [tag1, tag2, ...]
        prob_rating[pid_str] = int
    """
    import csv

    # --- Load problem metadata ---
    logger.info("Loading problem metadata...")
    prob_ids: Set[str] = set()
    prob_tags: Dict[str, List[str]] = {}
    prob_rating: Dict[str, int] = {}

    with open(PROBLEMS_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = row['id'].strip()  # e.g. "2220_B"
            prob_ids.add(pid)
            prob_ids.add(pid.replace('_', ''))  # also "2220B" for matching submissions

            tags = [t.strip() for t in row.get('tags', '').split(';') if t.strip()]
            prob_tags[pid] = tags

            r = row.get('rating', '')
            if r and r.strip():
                try:
                    prob_rating[pid] = int(r)
                except ValueError:
                    pass

    logger.info(f"  Loaded {len(prob_ids)} problems, {len(set(t for tags in prob_tags.values() for t in tags))} unique tags")

    # --- Filter submissions ---
    logger.info("Filtering submissions (this scans 829MB, takes ~30s)...")
    user_items: Dict[str, List[Tuple[str, int, int, List[str]]]] = defaultdict(list)
    # user_items[handle] = [(pid, timestamp, rating, tags), ...]

    total = 0
    ok_count = 0
    matched = 0

    with open(SUBMISSIONS_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            total += 1
            if total % 2_000_000 == 0:
                logger.info(f"  Scanned {total // 1_000_000}M rows...")

            if row['verdict'].strip() != 'OK':
                continue
            ok_count += 1

            pid = row['id_of_submission_task'].strip()
            if pid not in prob_ids:
                continue
            matched += 1

            handle = row['handle'].strip()
            ts = int(row['time'])
            rating = prob_rating.get(pid, 0)
            tags = prob_tags.get(pid, [])

            user_items[handle].append((pid, ts, rating, tags))

    logger.info(f"  Total: {total}, OK: {ok_count}, Matched: {matched}")

    # --- Filter users by min interactions ---
    qualified = {
        u: items
        for u, items in user_items.items()
        if len(set(pid for pid, _, _, _ in items)) >= min_user_interactions
    }
    logger.info(f"  Users with >= {min_user_interactions} distinct AC problems: {len(qualified)}")

    # Sort each user's items by timestamp and deduplicate (keep first AC)
    for u in qualified:
        seen = set()
        deduped = []
        for pid, ts, rating, tags in sorted(qualified[u], key=lambda x: x[1]):
            if pid not in seen:
                seen.add(pid)
                deduped.append((pid, ts, rating, tags))
        qualified[u] = deduped

    # Cap users if requested
    if max_users and len(qualified) > max_users:
        import random
        random.seed(42)
        keys = sorted(qualified.keys())
        selected = set(random.sample(keys, max_users))
        qualified = {k: v for k, v in qualified.items() if k in selected}
        logger.info(f"  Capped to {max_users} users (random seed=42)")

    return {
        'user_items': qualified,
        'prob_tags': prob_tags,
        'prob_rating': prob_rating,
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 2: Build interaction matrix + mappings
# ──────────────────────────────────────────────────────────────────────

def build_interaction_matrix(data: dict) -> dict:
    """
    Convert user_items dict to numeric indices.

    Returns dict with:
        user2id: {handle_str → int}
        id2user: {int → handle_str}
        prob2id: {pid_str → int}
        id2prob: {int → pid_str}
        X: list of (user_idx, item_idx) for all interactions
        user_items_idx: {user_idx → [(item_idx, ts), ...]}  sorted by time
        prob_tags: {pid_str → [tag, ...]}
        prob_rating: {pid_str → int}
        all_tags: sorted list of all unique tags
        n_users, n_items, n_interactions
    """
    user_items = data['user_items']

    # Build mappings
    all_users = sorted(user_items.keys())
    user2id = {u: i for i, u in enumerate(all_users)}
    id2user = {v: k for k, v in user2id.items()}

    all_probs = sorted(set(
        pid
        for items in user_items.values()
        for pid, _, _, _ in items
    ))
    prob2id = {p: i for i, p in enumerate(all_probs)}
    id2prob = {v: k for k, v in prob2id.items()}

    # Build interaction list
    X = []
    user_items_idx = defaultdict(list)

    for handle, items in user_items.items():
        u = user2id[handle]
        for pid, ts, rating, tags in items:
            p = prob2id.get(pid)
            if p is None:
                continue
            X.append((u, p))
            user_items_idx[u].append((p, ts))

    # Sort each user's items by timestamp
    for u in user_items_idx:
        user_items_idx[u].sort(key=lambda x: x[1])

    # Build sorted tag list
    all_tags = sorted(set(
        tag
        for tags in data['prob_tags'].values()
        for tag in tags
    ))

    n_users = len(user2id)
    n_items = len(prob2id)
    n_interactions = len(X)

    logger.info(f"Interaction matrix: {n_users} users × {n_items} items × {n_interactions} interactions")
    logger.info(f"  Avg interactions/user: {n_interactions / n_users:.1f}")
    logger.info(f"  Unique tags: {len(all_tags)}")

    return {
        'user2id': user2id,
        'id2user': id2user,
        'prob2id': prob2id,
        'id2prob': id2prob,
        'X': X,
        'user_items_idx': dict(user_items_idx),
        'prob_tags': data['prob_tags'],
        'prob_rating': data['prob_rating'],
        'all_tags': all_tags,
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 3: Build Knowledge Graph from tags
# ──────────────────────────────────────────────────────────────────────

def build_knowledge_graph(matrix: dict) -> dict:
    """
    Build KG from Codeforces tags:
      - Nodes: tags (topics) + problems + users
      - Edges: problem → tag (BELONGS_TO), user → problem (SOLVED)
      - Tag co-occurrence → SIMILAR_TO edges

    Returns dict:
        triples: list of (head, relation, tail) for KG embedding training
        entity2id, relation2id mappings
        tag_similarity: {(tag1,tag2): cooccurrence_count}
    """
    prob_tags = matrix['prob_tags']
    user_items = matrix['user_items_idx']
    id2prob = matrix['id2prob']
    all_tags = matrix['all_tags']

    # Entity: tag + problem + user
    entities = list(all_tags)
    tag2id = {t: i for i, t in enumerate(entities)}
    offset_p = len(entities)
    prob_entity2id = {pid: offset_p + i for i, pid in enumerate(sorted(prob_tags.keys()))}

    triples = []

    # BELONGS_TO: problem → tag
    for pid, tags in prob_tags.items():
        if pid not in prob_entity2id:
            continue
        p_eid = prob_entity2id[pid]
        for t in tags:
            t_eid = tag2id.get(t)
            if t_eid is not None:
                triples.append((p_eid, 0, t_eid))  # 0 = BELONGS_TO

    # SOLVED: user → problem
    offset_u = offset_p + len(prob_entity2id)
    for u_idx, items in user_items.items():
        u_eid = offset_u + u_idx
        for p_idx, _ in items:
            pid = id2prob.get(p_idx)
            if pid and pid in prob_entity2id:
                triples.append((u_eid, 1, prob_entity2id[pid]))  # 1 = SOLVED

    # SIMILAR_TO: tag co-occurrence in problems
    tag_cooccur = defaultdict(int)
    for pid, tags in prob_tags.items():
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = sorted([tags[i], tags[j]])
                tag_cooccur[(a, b)] += 1

    # Keep top co-occurring pairs as SIMILAR_TO edges
    threshold = np.percentile(list(tag_cooccur.values()), 50) if tag_cooccur else 0
    for (a, b), cnt in tag_cooccur.items():
        if cnt >= threshold:
            a_eid = tag2id[a]
            b_eid = tag2id[b]
            triples.append((a_eid, 2, b_eid))  # 2 = SIMILAR_TO
            triples.append((b_eid, 2, a_eid))

    # Relation names
    relation2id = {
        'BELONGS_TO': 0,
        'SOLVED': 1,
        'SIMILAR_TO': 2,
    }
    id2relation = {v: k for k, v in relation2id.items()}

    n_entities = offset_u + matrix['n_users']
    logger.info(f"KG: {n_entities} entities, {len(triples)} triples, {len(relation2id)} relations")
    logger.info(f"  Tags: {len(all_tags)}, Problems: {len(prob_entity2id)}, Users: {matrix['n_users']}")
    logger.info(f"  Tag co-occurrence edges: {sum(1 for _,r,_ in triples if r==2)} (threshold={threshold:.0f})")

    return {
        'triples': triples,
        'tag2id': tag2id,
        'prob_entity2id': prob_entity2id,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'n_entities': n_entities,
        'n_relations': len(relation2id),
        'tag_cooccur': dict(tag_cooccur),
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 4: Train/Test Splits
# ──────────────────────────────────────────────────────────────────────

def temporal_split(user_items_idx: dict, test_ratio: float = 0.2) -> Tuple[List, List]:
    """
    For each user: earliest (1-test_ratio) → train, latest → test.
    """
    X_train, X_test = [], []
    for u, items in user_items_idx.items():
        n = len(items)
        if n < 2:
            continue
        split = max(1, int(n * (1 - test_ratio)))
        for p, _ in items[:split]:
            X_train.append((u, p))
        for p, _ in items[split:]:
            X_test.append((u, p))
    logger.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test


def leave_one_out_split(user_items_idx: dict) -> Tuple[List, List]:
    """
    Leave-one-out: last interaction → test, rest → train.
    """
    X_train, X_test = [], []
    skipped = 0
    for u, items in user_items_idx.items():
        if len(items) < 2:
            skipped += 1
            continue
        for p, _ in items[:-1]:
            X_train.append((u, p))
        X_test.append((u, items[-1][0]))
    logger.info(f"LOO split: {len(X_train)} train, {len(X_test)} test"
                f" ({len(user_items_idx) - skipped} users, {skipped} skipped)")
    return X_train, X_test


# ──────────────────────────────────────────────────────────────────────
# STEP 5: Orchestrate full pipeline
# ──────────────────────────────────────────────────────────────────────

def load_codeforces_data(
    min_user_interactions: int = 5,
    max_users: int = None,
    split_type: str = 'loo',  # 'loo' or 'temporal'
    force_reload: bool = False,
) -> dict:
    """
    Full pipeline: load → filter → matrix → KG → split.
    Caches intermediate results to /home/proach/data/processed/.

    Returns dict with all data needed for experiments.
    """
    ensure_processed_dir()
    cache_path = os.path.join(
        PROCESSED_DIR,
        f'codeforces_u{max_users or "all"}_min{min_user_interactions}.pkl'
    )

    if os.path.exists(cache_path) and not force_reload:
        logger.info(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Step 1: Load & filter
    data = load_and_filter(
        min_user_interactions=min_user_interactions,
        max_users=max_users,
    )

    # Step 2: Build matrix
    matrix = build_interaction_matrix(data)

    # Step 3: Build KG
    kg = build_knowledge_graph(matrix)

    # Step 4: Split
    if split_type == 'loo':
        X_train, X_test = leave_one_out_split(matrix['user_items_idx'])
    else:
        X_train, X_test = temporal_split(matrix['user_items_idx'], test_ratio=0.2)

    result = {
        **matrix,
        'kg': kg,
        'X_train': X_train,
        'X_test': X_test,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    # Cache
    logger.info(f"Caching to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Done.")

    return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=None, help='Max users (None=all)')
    ap.add_argument('--min', type=int, default=5, help='Min interactions per user')
    ap.add_argument('--split', default='loo', choices=['loo', 'temporal'])
    ap.add_argument('--force', action='store_true', help='Force reload (ignore cache)')
    args = ap.parse_args()

    result = load_codeforces_data(
        min_user_interactions=args.min,
        max_users=args.users,
        split_type=args.split,
        force_reload=args.force,
    )

    print(f"\n=== Dataset Summary ===")
    print(f"  Users:            {result['n_users']}")
    print(f"  Items:            {result['n_items']}")
    print(f"  Interactions:     {result['n_interactions']}")
    print(f"  Train pairs:      {result['n_train']}")
    print(f"  Test pairs:       {result['n_test']}")
    print(f"  Tags:             {len(result['all_tags'])}")
    print(f"  KG entities:      {result['kg']['n_entities']}")
    print(f"  KG triples:       {len(result['kg']['triples'])}")
    print(f"  Avg items/user:   {result['n_interactions'] / result['n_users']:.1f}")
