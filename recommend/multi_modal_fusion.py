import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RecallCandidate:
    problem_id: str
    source: str
    score: float = 0.0
    reason: str = ""


@dataclass
class FusionResult:
    problem_id: str
    final_score: float
    recall_sources: List[str] = field(default_factory=list)
    reason: str = ""
    recall_score: float = 0.0
    rank_score: float = 0.0


class MultiModalFusionEngine:
    def __init__(self, model_dir='recommend_models'):
        self.model_dir = model_dir

        self.gnn_embeddings = None
        self.gnn_graph_data = None
        self.sequence_model = None
        self.sequence_data = None
        self.ranking_model = None
        self.ranking_data = None
        self.simple_model = None
        self.simple_mappings = None

        self._load_models()

    def _load_models(self):
        try:
            from recommend.gnn_recall import load_gnn_recall
            self.gnn_embeddings, self.gnn_graph_data = load_gnn_recall(self.model_dir)
            if self.gnn_embeddings:
                logger.info("GNN 召回模型加载成功")
        except Exception as e:
            logger.warning(f"GNN 召回模型加载失败: {e}")

        try:
            from recommend.sequence_recall import load_sequence_recall
            self.sequence_model, self.sequence_data = load_sequence_recall(self.model_dir)
            if self.sequence_model:
                logger.info("序列召回模型加载成功")
        except Exception as e:
            logger.warning(f"序列召回模型加载失败: {e}")

        try:
            from recommend.deepfm_ranking import load_deepfm
            self.ranking_model, self.ranking_data = load_deepfm(self.model_dir, use_xdeepfm=False)
            if self.ranking_model:
                logger.info("DeepFM 精排模型加载成功")
        except Exception as e:
            logger.warning(f"DeepFM 精排模型加载失败: {e}")

        try:
            import torch
            import pickle
            model_path = os.path.join(self.model_dir, '..', 'recommend_model.pt')
            data_path = os.path.join(self.model_dir, '..', 'recommend_data.pkl')
            if os.path.exists(model_path) and os.path.exists(data_path):
                from recommend.model import SimpleRecommender
                with open(data_path, 'rb') as f:
                    self.simple_mappings = pickle.load(f)
                self.simple_model = SimpleRecommender(
                    self.simple_mappings['num_users'],
                    self.simple_mappings['num_items'],
                    emb_dim=64
                )
                self.simple_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.simple_model.eval()
                logger.info("旧版 MLP 模型加载成功 (作为 fallback)")
        except Exception as e:
            logger.warning(f"旧版 MLP 模型加载失败: {e}")

    def gnn_recall(self, user_id: int, top_k: int = 50) -> List[RecallCandidate]:
        if self.gnn_embeddings is None or self.gnn_graph_data is None:
            return []
        from recommend.gnn_recall import gnn_recall
        results = gnn_recall(user_id, self.gnn_embeddings, self.gnn_graph_data, top_k)
        return [
            RecallCandidate(problem_id=pid, source='gnn', score=score,
                            reason="基于知识图谱图神经网络推荐")
            for pid, score in results
        ]

    def sequence_recall(self, user_id: int, top_k: int = 50) -> List[RecallCandidate]:
        if self.sequence_model is None or self.sequence_data is None:
            return []
        from recommend.sequence_recall import sequence_recall
        results = sequence_recall(user_id, self.sequence_model, self.sequence_data, top_k)
        return [
            RecallCandidate(problem_id=pid, source='sequence', score=score,
                            reason="基于您的做题序列智能推荐")
            for pid, score in results
        ]

    def aggregator_fusion(self, candidates: List[RecallCandidate],
                          weight_config: Dict[str, float] = None) -> List[RecallCandidate]:
        if weight_config is None:
            weight_config = {
                'graph_rule': 0.25,
                'graph_cf': 0.15,
                'gnn': 0.25,
                'sequence': 0.25,
                'hot': 0.10,
            }

        source_weights = defaultdict(float)
        for src, w in weight_config.items():
            source_weights[src] = w

        scored = defaultdict(lambda: {'score': 0.0, 'sources': [], 'reason': '', 'max_recall_score': 0.0})
        for c in candidates:
            w = source_weights.get(c.source, 0.1)
            entry = scored[c.problem_id]
            entry['score'] += c.score * w
            entry['sources'].append(c.source)
            entry['max_recall_score'] = max(entry['max_recall_score'], c.score)
            if c.reason:
                entry['reason'] = c.reason

        aggregated = []
        for pid, info in scored.items():
            diversity_bonus = len(set(info['sources'])) * 0.1
            final_score = info['score'] + diversity_bonus
            aggregated.append((pid, final_score, info['sources'], info['reason'], info['max_recall_score']))

        aggregated.sort(key=lambda x: x[1], reverse=True)
        return [
            RecallCandidate(
                problem_id=pid, source='fusion', score=score,
                reason=f"{reason} [召回源: {','.join(sources[:3])}]"
            )
            for pid, score, sources, reason, _ in aggregated
        ]

    def deepfm_rerank(self, user_id: int,
                      candidates: List[RecallCandidate]) -> List[RecallCandidate]:
        if self.ranking_model is None or self.ranking_data is None:
            return sorted(candidates, key=lambda x: x.score, reverse=True)

        from recommend.deepfm_ranking import deepfm_rank
        candidate_ids = [c.problem_id for c in candidates]
        ranked = deepfm_rank(user_id, candidate_ids, self.ranking_model, self.ranking_data)

        score_map = dict(ranked)
        for c in candidates:
            c.score = score_map.get(c.problem_id, c.score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def simple_rerank(self, user_id: int,
                      candidates: List[RecallCandidate]) -> List[RecallCandidate]:
        if self.simple_model is None or self.simple_mappings is None:
            return candidates

        import torch
        user_inner_id = self.simple_mappings['user2id'].get(user_id)
        if user_inner_id is None:
            return sorted(candidates, key=lambda x: x.score, reverse=True)

        candidate_ids = []
        candidate_inner_ids = []
        for c in candidates:
            inner_id = self.simple_mappings['prob2id'].get(c.problem_id)
            if inner_id is not None:
                candidate_ids.append(c.problem_id)
                candidate_inner_ids.append(inner_id)

        if not candidate_inner_ids:
            return sorted(candidates, key=lambda x: x.score, reverse=True)

        user_tensor = torch.tensor([user_inner_id] * len(candidate_inner_ids), dtype=torch.long)
        item_tensor = torch.tensor(candidate_inner_ids, dtype=torch.long)
        with torch.no_grad():
            scores = self.simple_model(user_tensor, item_tensor).numpy().flatten()

        score_map = dict(zip(candidate_ids, scores))
        for c in candidates:
            c.score = score_map.get(c.problem_id, c.score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)


_engine_instance = None


def get_fusion_engine(model_dir='recommend_models') -> MultiModalFusionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MultiModalFusionEngine(model_dir)
    return _engine_instance
