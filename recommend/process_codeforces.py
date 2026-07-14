"""
Standalone script to process Codeforces CSV data on the HOST machine,
and save as pickle for the container to load.

Usage: python3 process_codeforces.py [--users N] [--min M]
"""

import csv
import pickle
import random
import os
from collections import defaultdict

DATA_DIR = '/home/proach/publicData'
OUTPUT_DIR = '/home/proach/publicData/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROBLEMS_CSV = os.path.join(DATA_DIR, 'codeforces_problems.csv')
SUBMISSIONS_CSV = os.path.join(DATA_DIR, 'usersCodeforcesSubmissionsEnd2024.csv')


def process(max_users=None, min_interactions=5):
    # 1. Load problem metadata
    print("Loading problems...")
    prob_ids = set()
    prob_tags = {}
    prob_rating = {}
    with open(PROBLEMS_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = row['id'].strip().replace('_', '')  # normalize: "2220_B" → "2220B"
            prob_ids.add(pid)
            tags = [t.strip() for t in row.get('tags', '').split(';') if t.strip()]
            prob_tags[pid] = tags
            r = row.get('rating', '')
            if r and r.strip():
                try:
                    prob_rating[pid] = int(r)
                except ValueError:
                    pass
    print(f"  {len(prob_ids)} problem IDs, {len(set(t for tags in prob_tags.values() for t in tags))} tags")

    # 2. Filter submissions
    print("Filtering submissions (scanning 829MB, ~30s)...")
    user_items = defaultdict(list)
    total = ok_count = matched = 0
    with open(SUBMISSIONS_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            total += 1
            if total % 2_000_000 == 0:
                print(f"  {total // 1_000_000}M rows...")
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
            user_items[handle].append((pid, ts, rating, tuple(tags)))
    print(f"  Total={total}, OK={ok_count}, Matched={matched}")

    # 3. Filter users, deduplicate, sort by time
    print("Filtering users...")
    qualified = {}
    for u, items in user_items.items():
        # unique problems
        unique = {}
        for pid, ts, rating, tags in items:
            if pid not in unique or unique[pid][0] > ts:
                unique[pid] = (ts, rating, tags)
        if len(unique) >= min_interactions:
            qualified[u] = sorted(
                [(pid, ts, rating, tags) for pid, (ts, rating, tags) in unique.items()],
                key=lambda x: x[1]
            )
    print(f"  {len(qualified)} users with >= {min_interactions} problems")

    if max_users and len(qualified) > max_users:
        random.seed(42)
        keys = sorted(qualified.keys())
        selected = set(random.sample(keys, max_users))
        qualified = {k: v for k, v in qualified.items() if k in selected}
        print(f"  Capped to {max_users} users")

    # 4. Build numeric mappings
    print("Building mappings...")
    all_users = sorted(qualified.keys())
    user2id = {u: i for i, u in enumerate(all_users)}
    all_probs = sorted(set(
        pid for items in qualified.values() for pid, _, _, _ in items
    ))
    prob2id = {p: i for i, p in enumerate(all_probs)}
    id2prob = {v: k for k, v in prob2id.items()}
    id2user = {v: k for k, v in user2id.items()}

    X = []
    user_items_idx = {}
    for u, items in qualified.items():
        ui = user2id[u]
        idx_items = []
        for pid, ts, rating, tags in items:
            p = prob2id[pid]
            X.append((ui, p))
            idx_items.append((p, ts, rating, list(tags)))
        user_items_idx[ui] = idx_items

    all_tags = sorted(set(t for tags in prob_tags.values() for t in tags))

    n_users = len(user2id)
    n_items = len(prob2id)
    print(f"  {n_users} users x {n_items} items x {len(X)} interactions")
    print(f"  Avg: {len(X)/n_users:.1f} per user")

    # 5. LOO split
    print("Building splits...")
    X_train, X_test = [], []
    skipped = 0
    for u, items in user_items_idx.items():
        if len(items) < 2:
            skipped += 1
            continue
        for p, _, _, _ in items[:-1]:
            X_train.append((u, p))
        X_test.append((u, items[-1][0]))
    print(f"  LOO: {len(X_train)} train, {len(X_test)} test"
          f" ({n_users - skipped} users, {skipped} skipped)")

    # 6. Build KG from tags
    print("Building KG...")
    tag_cooccur = defaultdict(int)
    for pid, taglist in prob_tags.items():
        for i in range(len(taglist)):
            for j in range(i + 1, len(taglist)):
                a, b = sorted([taglist[i], taglist[j]])
                tag_cooccur[(a, b)] += 1

    # Normalize tag similarity
    vals = list(tag_cooccur.values())
    threshold = sorted(vals)[len(vals) // 2] if vals else 0
    tag_sim_edges = []
    for (a, b), cnt in tag_cooccur.items():
        if cnt >= threshold:
            tag_sim_edges.append((a, b, cnt))

    # Tag2id
    tag2id = {t: i for i, t in enumerate(all_tags)}

    print(f"  {len(all_tags)} tags, {len(tag_sim_edges)} similarity edges"
          f" (threshold={threshold})")

    # 7. Save
    result = {
        'user2id': user2id,
        'id2user': id2user,
        'prob2id': prob2id,
        'id2prob': id2prob,
        'X': X,
        'user_items_idx': user_items_idx,
        'prob_tags': prob_tags,
        'prob_rating': prob_rating,
        'all_tags': all_tags,
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': len(X),
        'X_train': X_train,
        'X_test': X_test,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'tag2id': tag2id,
        'tag_cooccur': dict(tag_cooccur),
        'tag_sim_edges': tag_sim_edges,
    }

    basename = f'codeforces_u{max_users or "all"}_min{min_interactions}'
    cache_path = os.path.join(OUTPUT_DIR, basename + '.pkl')
    print(f"Saving to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    stat_path = os.path.join(OUTPUT_DIR, basename + '_stats.txt')
    with open(stat_path, 'w') as f:
        f.write(f"Users: {n_users}\n")
        f.write(f"Items: {n_items}\n")
        f.write(f"Interactions: {len(X)}\n")
        f.write(f"Train pairs: {len(X_train)}\n")
        f.write(f"Test pairs: {len(X_test)}\n")
        f.write(f"Tags: {len(all_tags)}\n")
        f.write(f"Avg items/user: {len(X)/n_users:.1f}\n")

    print(f"Stats saved to {stat_path}")
    print("Done!")
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=None)
    ap.add_argument('--min', type=int, default=5)
    args = ap.parse_args()
    process(max_users=args.users, min_interactions=args.min)
