import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django
django.setup()

import torch
import numpy as np
from submission.models import Submission, JudgeStatus
from problem.models import Problem
from account.models import User
import pickle

def export():
    users = list(User.objects.values_list('id', flat=True).order_by('id'))
    user2id = {uid: i for i, uid in enumerate(users)}
    problems = list(Problem.objects.values_list('id', flat=True).order_by('id'))
    prob2id = {pid: i for i, pid in enumerate(problems)}
    
    pos_pairs = Submission.objects.filter(result=JudgeStatus.ACCEPTED).values_list('user_id', 'problem_id').distinct()
    pos = [(user2id[u], prob2id[p]) for u,p in pos_pairs if u in user2id and p in prob2id]
    
    # 负采样
    neg_pairs = []
    total_items = len(problems)
    for u in user2id.values():
        user_pos = set(p for (uid,p) in pos if uid == u)
        available = [i for i in range(total_items) if i not in user_pos]
        num_neg = min(len(user_pos), 4) if user_pos else 4
        if len(available) >= num_neg:
            neg_samples = np.random.choice(available, size=num_neg, replace=False)
            for p in neg_samples:
                neg_pairs.append((u, p))
    
    X = pos + neg_pairs
    y = [1]*len(pos) + [0]*len(neg_pairs)
    
    # 手动划分训练/测试
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train = [X[i] for i in indices[:split]]
    y_train = [y[i] for i in indices[:split]]
    X_test = [X[i] for i in indices[split:]]
    y_test = [y[i] for i in indices[split:]]
    
    with open('/data/recommend_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'num_users': len(users), 'num_items': len(problems),
            'user2id': user2id, 'prob2id': prob2id
        }, f)
    print(f"数据导出完成。正样本:{len(pos)}, 负样本:{len(neg_pairs)}")

if __name__ == '__main__':
    export()