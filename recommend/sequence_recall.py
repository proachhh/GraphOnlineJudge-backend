import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django
django.setup()

from submission.models import Submission, JudgeStatus
from problem.models import Problem, ProblemTag


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SequenceTransformer(nn.Module):
    def __init__(self, num_problems, num_topics, num_results,
                 d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.problem_emb = nn.Embedding(num_problems, d_model)
        self.topic_emb = nn.Embedding(num_topics, d_model)
        self.result_emb = nn.Embedding(num_results, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_problems)
        self.dropout = nn.Dropout(dropout)

    def forward(self, problem_seq, topic_seq, result_seq):
        p_emb = self.problem_emb(problem_seq)
        t_emb = self.topic_emb(topic_seq)
        r_emb = self.result_emb(result_seq)
        x = self.fusion(torch.cat([p_emb, t_emb, r_emb], dim=-1))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        logits = self.output_layer(x)
        return logits

    def get_sequence_embedding(self, problem_seq, topic_seq, result_seq):
        p_emb = self.problem_emb(problem_seq)
        t_emb = self.topic_emb(topic_seq)
        r_emb = self.result_emb(result_seq)
        x = self.fusion(torch.cat([p_emb, t_emb, r_emb], dim=-1))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x[:, -1, :]


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, num_problems):
        self.sequences = sequences
        self.targets = targets
        self.num_problems = num_problems

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        problem_seq = torch.tensor([s[0] for s in seq], dtype=torch.long)
        topic_seq = torch.tensor([s[1] for s in seq], dtype=torch.long)
        result_seq = torch.tensor([s[2] for s in seq], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return problem_seq, topic_seq, result_seq, target


class SequenceRecallTrainer:
    def __init__(self, save_dir='recommend_models', seq_len=20):
        self.save_dir = save_dir
        self.seq_len = seq_len
        os.makedirs(save_dir, exist_ok=True)

    def export_sequence_data(self):
        users = list(Submission.objects.values_list('user_id', flat=True).distinct().order_by('user_id'))
        problems = list(Problem.objects.values_list('id', flat=True).order_by('id'))

        prob2idx = {pid: i for i, pid in enumerate(problems)}
        idx2prob = {i: pid for pid, i in prob2idx.items()}

        tag_names = list(ProblemTag.objects.values_list('name', flat=True).distinct().order_by('name'))
        topic2idx = {tn: i for i, tn in enumerate(tag_names)}

        result2idx = {
            JudgeStatus.ACCEPTED: 0,
            JudgeStatus.WRONG_ANSWER: 1,
            JudgeStatus.CPU_TIME_LIMIT_EXCEEDED: 2,
            JudgeStatus.REAL_TIME_LIMIT_EXCEEDED: 2,
            JudgeStatus.MEMORY_LIMIT_EXCEEDED: 3,
            JudgeStatus.RUNTIME_ERROR: 4,
            JudgeStatus.COMPILE_ERROR: 5,
        }

        problem_topics = {}
        for p in Problem.objects.prefetch_related('tags').all():
            if p.id in prob2idx:
                tag_list = list(p.tags.values_list('name', flat=True))
                if tag_list:
                    problem_topics[prob2idx[p.id]] = topic2idx.get(tag_list[0], 0)
                else:
                    problem_topics[prob2idx[p.id]] = 0

        sequences = []
        targets = []

        for user_id in users:
            subs = Submission.objects.filter(user_id=user_id).order_by('create_time')
            user_seq = []
            for sub in subs:
                pid_idx = prob2idx.get(sub.problem_id)
                if pid_idx is None:
                    continue
                topic_idx = problem_topics.get(pid_idx, 0)
                result_idx = result2idx.get(sub.result, 0)
                user_seq.append((pid_idx, topic_idx, result_idx))

            for i in range(self.seq_len, len(user_seq)):
                seq = user_seq[i - self.seq_len:i]
                target = user_seq[i][0]
                sequences.append(seq)
                targets.append(target)

        data = {
            'sequences': sequences,
            'targets': targets,
            'num_problems': len(problems),
            'num_topics': len(tag_names),
            'num_results': 6,
            'prob2idx': prob2idx,
            'idx2prob': idx2prob,
            'topic2idx': topic2idx,
            'user_ids': users,
        }

        with open(os.path.join(self.save_dir, 'sequence_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"序列数据导出完成: {len(sequences)} 条训练样本, {len(problems)} 题目, {len(tag_names)} 知识点")
        return data

    def train(self, epochs=30, batch_size=64, lr=0.001):
        data_path = os.path.join(self.save_dir, 'sequence_data.pkl')
        if not os.path.exists(data_path):
            print("未找到序列数据，开始导出...")
            self.export_sequence_data()

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        num_problems = data['num_problems']
        num_topics = max(data['num_topics'], 1)
        num_results = data['num_results']

        split = int(0.8 * len(data['sequences']))
        train_seqs = data['sequences'][:split]
        train_targets = data['targets'][:split]
        test_seqs = data['sequences'][split:]
        test_targets = data['targets'][split:]

        train_dataset = SequenceDataset(train_seqs, train_targets, num_problems)
        test_dataset = SequenceDataset(test_seqs, test_targets, num_problems)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = SequenceTransformer(num_problems, num_topics, num_results,
                                     d_model=128, nhead=4, num_layers=2)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for p_seq, t_seq, r_seq, targets in train_loader:
                optimizer.zero_grad()
                logits = model(p_seq, t_seq, r_seq)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for p_seq, t_seq, r_seq, targets in test_loader:
                        logits = model(p_seq, t_seq, r_seq)
                        pred = logits.argmax(dim=1)
                        correct += (pred == targets).sum().item()
                        total += targets.size(0)
                acc = correct / total if total > 0 else 0
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc@1: {acc:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(self.save_dir, 'sequence_recall.pt'))

        torch.save(model.state_dict(), os.path.join(self.save_dir, 'sequence_recall.pt'))
        print("序列召回模型已保存")
        return model


def load_sequence_recall(model_dir='recommend_models'):
    model_path = os.path.join(model_dir, 'sequence_recall.pt')
    data_path = os.path.join(model_dir, 'sequence_data.pkl')
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    model = SequenceTransformer(
        data['num_problems'], max(data['num_topics'], 1), data['num_results'],
        d_model=128, nhead=4, num_layers=2
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, data


def sequence_recall(user_id, model, data, top_k=50):
    result2idx = {
        0: 0, -1: 1, 1: 2, 2: 2, 3: 3, 4: 4, -2: 5,
    }

    subs = Submission.objects.filter(user_id=user_id).order_by('create_time')[:50]
    seq_len = 20

    problem_topics = {}
    for p in Problem.objects.prefetch_related('tags').all():
        if p.id in data['prob2idx']:
            tag_list = list(p.tags.values_list('name', flat=True))
            problem_topics[data['prob2idx'][p.id]] = data['topic2idx'].get(tag_list[0], 0) if tag_list else 0

    user_seq = []
    for sub in subs:
        pid_idx = data['prob2idx'].get(sub.problem_id)
        if pid_idx is None:
            continue
        topic_idx = problem_topics.get(pid_idx, 0)
        result_idx = result2idx.get(sub.result, 0)
        user_seq.append((pid_idx, topic_idx, result_idx))

    if len(user_seq) < 2:
        return []

    input_seq = user_seq[-seq_len:] if len(user_seq) > seq_len else user_seq

    p_seq = torch.tensor([[s[0] for s in input_seq]], dtype=torch.long)
    t_seq = torch.tensor([[s[1] for s in input_seq]], dtype=torch.long)
    r_seq = torch.tensor([[s[2] for s in input_seq]], dtype=torch.long)

    with torch.no_grad():
        logits = model(p_seq, t_seq, r_seq).squeeze(0)

    done_pids = set(s[0] for s in user_seq)
    scores, indices = torch.topk(logits, k=min(top_k + len(done_pids), len(logits)))

    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        if idx in done_pids:
            continue
        pid = data['idx2prob'].get(idx)
        if pid is not None:
            results.append((pid, score))
        if len(results) >= top_k:
            break

    return results


if __name__ == '__main__':
    trainer = SequenceRecallTrainer()
    trainer.export_sequence_data()
    trainer.train(epochs=30)
