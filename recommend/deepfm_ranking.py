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

from submission.models import Submission, JudgeStatus
from problem.models import Problem, ProblemTag
from account.models import User


class FeatureConfig:
    NUMERICAL_FEATURES = [
        'user_total_submissions', 'user_ac_rate', 'user_pass_rate',
        'problem_accepted_number', 'problem_submission_number', 'problem_pass_rate',
        'problem_difficulty_code',
        'user_topic_mastery', 'user_topic_attempts',
        'tag_overlap_count', 'time_since_last_submission',
    ]
    CATEGORICAL_FEATURES = [
        'user_id_hash', 'problem_tag_hash',
    ]
    EMB_DIM_PER_CAT = 16
    TOTAL_NUM_CATS = 1000


class FMBlock(nn.Module):
    def __init__(self, num_features, embed_dim=16):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.V = nn.Parameter(torch.randn(num_features, embed_dim) * 0.01)

    def forward(self, x):
        linear_part = self.linear(x)
        interactions = torch.mm(x, self.V)
        sum_square = torch.pow(interactions.sum(dim=1, keepdim=True), 2)
        square_sum = torch.pow(interactions, 2).sum(dim=1, keepdim=True)
        fm_part = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
        return linear_part + fm_part


class CINBlock(nn.Module):
    """xDeepFM CIN — 3D input, simple broadcasting, no expand trick"""
    def __init__(self, num_fields, layer_dims=(128, 128)):
        super().__init__()
        prev_h = num_fields  # H_0 = F
        self.conv_layers = nn.ModuleList()
        for dim in layer_dims:
            self.conv_layers.append(
                nn.Conv1d(in_channels=prev_h * num_fields, out_channels=dim, kernel_size=1)
            )
            prev_h = dim

    def forward(self, x_embedded):
        # x_embedded: [B, nf, D]
        B, nf, D = x_embedded.shape
        x0 = x_embedded                # [B, nf, D]
        xk = x_embedded                # [B, H_0=nf, D]
        results = [xk.sum(dim=1)]      # [B, D]

        for conv in self.conv_layers:
            # Interaction: z[h,f] = xk_h ⊙ x0_f  →  [B, H_k, nf, D]
            z = xk.unsqueeze(2) * x0.unsqueeze(1)
            H_k = z.size(1)
            z = z.reshape(B, H_k * nf, D)  # [B, H_k*nf, D]
            xk = conv(z)                   # [B, H_{k+1}, D]
            xk = F.relu(xk)
            results.append(xk.sum(dim=1))  # [B, D]

        return torch.cat(results, dim=1)   # [B, (L+1)*D]


class DeepFM(nn.Module):
    def __init__(self, num_numerical, num_categorical, cat_cardinality,
                 embed_dim=16, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        self.total_fields = num_numerical + num_categorical

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, embed_dim) for _ in range(num_categorical)
        ])

        self.num_linear = nn.Linear(num_numerical, embed_dim * num_numerical)

        self.fm = FMBlock(self.total_fields * embed_dim, embed_dim)

        dnn_input_dim = self.total_fields * embed_dim
        dnn_layers = []
        prev_dim = dnn_input_dim
        for h_dim in hidden_dims:
            dnn_layers.append(nn.Linear(prev_dim, h_dim))
            dnn_layers.append(nn.BatchNorm1d(h_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, numerical, categorical):
        batch_size = numerical.size(0)

        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(categorical[:, i]))

        num_expanded = numerical.unsqueeze(2).expand(-1, -1, self.embed_dim)
        num_embs = [num_expanded[:, i, :] for i in range(self.num_numerical)]

        all_fields = num_embs + cat_embs
        x_flat = torch.cat([f.reshape(batch_size, -1) for f in all_fields], dim=1)

        fm_out = self.fm(x_flat)
        dnn_out = self.dnn(x_flat)
        score = torch.sigmoid(fm_out + dnn_out).squeeze(-1)
        return score


class xDeepFM(nn.Module):
    def __init__(self, num_numerical, num_categorical, cat_cardinality,
                 embed_dim=16, hidden_dims=(256, 128, 64), cin_dims=(128, 128), dropout=0.2):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        self.total_fields = num_numerical + num_categorical

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, embed_dim) for _ in range(num_categorical)
        ])

        self.fm = FMBlock(self.total_fields * embed_dim, embed_dim)

        self.cin = CINBlock(self.total_fields, cin_dims)
        cin_out_dim = (len(cin_dims) + 1) * embed_dim

        dnn_input_dim = self.total_fields * embed_dim
        dnn_layers = []
        prev_dim = dnn_input_dim
        for h_dim in hidden_dims:
            dnn_layers.append(nn.Linear(prev_dim, h_dim))
            dnn_layers.append(nn.BatchNorm1d(h_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        self.output_linear = nn.Linear(cin_out_dim + 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, numerical, categorical):
        batch_size = numerical.size(0)

        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(categorical[:, i]))

        num_expanded = numerical.unsqueeze(2).expand(-1, -1, self.embed_dim)
        num_embs = [num_expanded[:, i, :] for i in range(self.num_numerical)]

        all_fields = num_embs + cat_embs
        x_flat = torch.cat([f.reshape(batch_size, -1) for f in all_fields], dim=1)
        x_stacked = torch.stack(all_fields, dim=1)

        fm_out = self.fm(x_flat)
        cin_out = self.cin(x_stacked)
        dnn_out = self.dnn(x_flat)

        combined = torch.cat([cin_out, dnn_out], dim=1)
        score = torch.sigmoid(self.output_linear(combined)).squeeze(-1)
        return score


class FeatureBuilder:
    def __init__(self):
        self.user_stats = {}
        self.problem_stats = {}

    def build_user_features(self, user_id):
        if user_id in self.user_stats:
            return self.user_stats[user_id]

        submissions = Submission.objects.filter(user_id=user_id)
        total = submissions.count()
        ac_count = submissions.filter(result=JudgeStatus.ACCEPTED).count()
        ac_rate = ac_count / total if total > 0 else 0

        features = {
            'user_total_submissions': np.log1p(total),
            'user_ac_rate': ac_rate,
            'user_pass_rate': ac_rate,
        }
        self.user_stats[user_id] = features
        return features

    def build_problem_features(self, problem_id):
        if problem_id in self.problem_stats:
            return self.problem_stats[problem_id]

        try:
            p = Problem.objects.get(id=problem_id)
            total_sub = p.submission_number
            ac_num = p.accepted_number
            pass_rate = ac_num / total_sub if total_sub > 0 else 0
            diff_map = {'Low': 0, 'Mid': 1, 'High': 2}
            features = {
                'problem_accepted_number': np.log1p(ac_num),
                'problem_submission_number': np.log1p(total_sub),
                'problem_pass_rate': pass_rate,
                'problem_difficulty_code': diff_map.get(p.difficulty, 1),
            }
        except Problem.DoesNotExist:
            features = {
                'problem_accepted_number': 0,
                'problem_submission_number': 0,
                'problem_pass_rate': 0,
                'problem_difficulty_code': 1,
            }
        self.problem_stats[problem_id] = features
        return features

    def build_cross_features(self, user_id, problem_id):
        user_tags = set(ProblemTag.objects.filter(
            problem__submission__user_id=user_id,
            problem__submission__result=JudgeStatus.ACCEPTED
        ).values_list('name', flat=True).distinct())

        try:
            prob_tags = set(ProblemTag.objects.filter(
                problem_id=problem_id
            ).values_list('name', flat=True))
        except Exception:
            prob_tags = set()

        tag_overlap = len(user_tags & prob_tags)

        last_sub = Submission.objects.filter(
            user_id=user_id
        ).order_by('-create_time').first()
        from django.utils import timezone
        time_since = 0
        if last_sub:
            delta = timezone.now() - last_sub.create_time
            time_since = delta.total_seconds() / 3600.0

        topic_mastery = 0.0
        topic_attempts = 0
        if prob_tags:
            submissions_for_tags = Submission.objects.filter(
                user_id=user_id,
                problem__tags__name__in=prob_tags
            )
            topic_attempts = submissions_for_tags.count()
            topic_ac = submissions_for_tags.filter(result=JudgeStatus.ACCEPTED).count()
            topic_mastery = topic_ac / topic_attempts if topic_attempts > 0 else 0

        features = {
            'tag_overlap_count': float(tag_overlap),
            'user_topic_mastery': topic_mastery,
            'user_topic_attempts': np.log1p(topic_attempts),
            'time_since_last_submission': np.log1p(time_since),
        }
        return features

    def build_sample(self, user_id, problem_id, label):
        u_feat = self.build_user_features(user_id)
        p_feat = self.build_problem_features(problem_id)
        cross_feat = self.build_cross_features(user_id, problem_id)
        return {**u_feat, **p_feat, **cross_feat, 'label': label}

    def build_numerical_vector(self, sample):
        feat_names = FeatureConfig.NUMERICAL_FEATURES
        return np.array([sample.get(f, 0.0) for f in feat_names], dtype=np.float32)

    def build_categorical_vector(self, sample, max_cats=1000):
        uid_hash = abs(hash(str(sample.get('user_id_hash', 0)))) % max_cats
        tag_hash = abs(hash(str(sample.get('problem_tag_hash', 0)))) % max_cats
        return np.array([uid_hash, tag_hash], dtype=np.int64)


class DeepFMTrainer:
    def __init__(self, save_dir='recommend_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.feature_builder = FeatureBuilder()

    def export_ranking_data(self, max_samples=50000):
        pos_pairs = list(Submission.objects.filter(
            result=JudgeStatus.ACCEPTED
        ).values_list('user_id', 'problem_id').distinct()[:max_samples])

        users = list(set(uid for uid, _ in pos_pairs))
        problems = list(set(pid for _, pid in pos_pairs))

        neg_pairs = []
        for uid in users[:500]:
            user_pos = set(pid for u, pid in pos_pairs if u == uid)
            available = [pid for pid in problems[:200] if pid not in user_pos]
            num_neg = min(len(user_pos), 3) if user_pos else 3
            if len(available) >= num_neg:
                neg_samples = np.random.choice(available, size=num_neg, replace=False)
                for pid in neg_samples:
                    neg_pairs.append((uid, pid))

        print(f"正样本: {len(pos_pairs)}, 负样本: {len(neg_pairs)}")

        samples = []
        for uid, pid in pos_pairs:
            sample = self.feature_builder.build_sample(uid, pid, 1)
            sample['user_id_hash'] = uid
            sample['problem_tag_hash'] = pid
            samples.append(sample)

        for uid, pid in neg_pairs:
            sample = self.feature_builder.build_sample(uid, pid, 0)
            sample['user_id_hash'] = uid
            sample['problem_tag_hash'] = pid
            samples.append(sample)

        X_numerical = np.array([self.feature_builder.build_numerical_vector(s) for s in samples])
        X_categorical = np.array([self.feature_builder.build_categorical_vector(s) for s in samples])
        y = np.array([s['label'] for s in samples], dtype=np.float32)

        indices = np.random.permutation(len(samples))
        split = int(0.8 * len(samples))
        train_idx = indices[:split]
        test_idx = indices[split:]

        data = {
            'X_train_num': X_numerical[train_idx],
            'X_train_cat': X_categorical[train_idx],
            'y_train': y[train_idx],
            'X_test_num': X_numerical[test_idx],
            'X_test_cat': X_categorical[test_idx],
            'y_test': y[test_idx],
            'num_numerical': len(FeatureConfig.NUMERICAL_FEATURES),
            'num_categorical': len(FeatureConfig.CATEGORICAL_FEATURES),
            'cat_cardinality': FeatureConfig.TOTAL_NUM_CATS,
            'feature_names': FeatureConfig.NUMERICAL_FEATURES + FeatureConfig.CATEGORICAL_FEATURES,
        }

        with open(os.path.join(self.save_dir, 'ranking_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"精排数据导出完成: {len(samples)} 条样本")
        return data

    def train_deepfm(self, epochs=30, batch_size=256, lr=0.001, use_xdeepfm=False):
        data_path = os.path.join(self.save_dir, 'ranking_data.pkl')
        if not os.path.exists(data_path):
            print("未找到精排数据，开始导出...")
            self.export_ranking_data()

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        X_train_num = torch.tensor(data['X_train_num'], dtype=torch.float)
        X_train_cat = torch.tensor(data['X_train_cat'], dtype=torch.long)
        y_train = torch.tensor(data['y_train'], dtype=torch.float)
        X_test_num = torch.tensor(data['X_test_num'], dtype=torch.float)
        X_test_cat = torch.tensor(data['X_test_cat'], dtype=torch.long)
        y_test = torch.tensor(data['y_test'], dtype=torch.float)

        if use_xdeepfm:
            model = xDeepFM(
                data['num_numerical'], data['num_categorical'], data['cat_cardinality'],
                hidden_dims=(256, 128, 64), cin_dims=(128, 128)
            )
            model_name = 'xdeepfm'
        else:
            model = DeepFM(
                data['num_numerical'], data['num_categorical'], data['cat_cardinality'],
                hidden_dims=(256, 128, 64)
            )
            model_name = 'deepfm'

        optimizer = optim.Adam(model.parameters(), lr=lr)
        n = len(X_train_num)
        best_auc = 0

        for epoch in range(epochs):
            model.train()
            idx = np.random.permutation(n)
            total_loss = 0
            batches = 0
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                num_batch = X_train_num[batch_idx]
                cat_batch = X_train_cat[batch_idx]
                y_batch = y_train[batch_idx]
                optimizer.zero_grad()
                pred = model(num_batch, cat_batch)
                loss = F.binary_cross_entropy(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    pred_test = model(X_test_num, X_test_cat).numpy()
                    auc = self._compute_auc(y_test.numpy(), pred_test)
                print(f"Epoch {epoch+1}, Loss: {total_loss/batches:.4f}, Test AUC: {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), os.path.join(self.save_dir, f'{model_name}_ranking.pt'))

        torch.save(model.state_dict(), os.path.join(self.save_dir, f'{model_name}_ranking.pt'))
        print(f"{model_name.upper()} 模型已保存, Best AUC: {best_auc:.4f}")
        return model

    def _compute_auc(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)
        if pos == 0 or neg == 0:
            return 0.5
        tp, fp, auc_val = 0, 0, 0
        for idx in order:
            if y_true[idx] == 1:
                tp += 1
            else:
                fp += 1
                auc_val += tp
        return auc_val / (pos * neg)


def load_deepfm(model_dir='recommend_models', use_xdeepfm=False):
    model_name = 'xdeepfm' if use_xdeepfm else 'deepfm'
    model_path = os.path.join(model_dir, f'{model_name}_ranking.pt')
    data_path = os.path.join(model_dir, 'ranking_data.pkl')

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if use_xdeepfm:
        model = xDeepFM(data['num_numerical'], data['num_categorical'], data['cat_cardinality'])
    else:
        model = DeepFM(data['num_numerical'], data['num_categorical'], data['cat_cardinality'])

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, data


def deepfm_rank(user_id, candidate_problem_ids, model, data):
    builder = FeatureBuilder()
    numerical_vecs = []
    categorical_vecs = []
    for pid in candidate_problem_ids:
        sample = builder.build_sample(user_id, pid, 0)
        sample['user_id_hash'] = user_id
        sample['problem_tag_hash'] = pid
        numerical_vecs.append(builder.build_numerical_vector(sample))
        categorical_vecs.append(builder.build_categorical_vector(sample))

    if not numerical_vecs:
        return []

    X_num = torch.tensor(np.array(numerical_vecs), dtype=torch.float)
    X_cat = torch.tensor(np.array(categorical_vecs), dtype=torch.long)

    with torch.no_grad():
        scores = model(X_num, X_cat).numpy()

    scored = list(zip(candidate_problem_ids, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


if __name__ == '__main__':
    trainer = DeepFMTrainer()
    trainer.export_ranking_data()
    trainer.train_deepfm(epochs=30, use_xdeepfm=False)
