import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

_fusion_engine = None


def _get_fusion_engine():
    global _fusion_engine
    if _fusion_engine is not None:
        return _fusion_engine
    try:
        import os
        from django.conf import settings
        model_dir = os.path.join(str(settings.BASE_DIR), 'recommend_models')
        from recommend.multi_modal_fusion import MultiModalFusionEngine
        _fusion_engine = MultiModalFusionEngine(model_dir)
        logger.info("多模态融合引擎初始化成功")
    except Exception as e:
        logger.warning(f"多模态融合引擎初始化失败: {e}")
    return _fusion_engine


def _get_graph_recommendations(username: str, limit: int = 20) -> List[Dict]:
    client = neo4j_client
    recs = []
    seen_ids = set()

    def add_rec(problem_id, _id, title, reason, score):
        if problem_id not in seen_ids:
            seen_ids.add(problem_id)
            recs.append({'id': _id or problem_id, '_id': _id or problem_id, 'title': title, 'reason': reason, 'score': score})

    prereq_query = """
    MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)-[:BELONGS_TO]->(t:Topic)
    WHERE s.result = '0'
    WITH u, t, count(DISTINCT s) AS ac_count
    WHERE ac_count >= 2
    MATCH (t)-[:PREREQUISITE_OF]->(next_topic:Topic)
    MATCH (next_topic)<-[:BELONGS_TO]-(rec:Problem)
    WHERE NOT EXISTS { MATCH (u)-[:SUBMITTED]->(:Submission)-[:FOR]->(rec) }
    RETURN DISTINCT rec.problem_id AS id, rec._id AS display_id, rec.title AS title,
           t.name AS mastered_topic, next_topic.name AS next_topic,
           rec.accepted_number AS ac_num
    ORDER BY ac_num DESC
    LIMIT 10
    """
    try:
        result = client.run_query(prereq_query, {'username': username})
        for r in result:
            add_rec(r['id'], r.get('display_id', ''), r.get('title', ''), f"您已掌握「{r['mastered_topic']}」，推荐学习「{r['next_topic']}」", score=90)
    except Exception as e:
        logger.error(f"前置知识点推荐查询失败: {e}")

    weak_query = """
    MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)-[:BELONGS_TO]->(t:Topic)
    WITH u, t, count(s) AS total,
         sum(CASE WHEN s.result = '0' THEN 1 ELSE 0 END) AS ac_count
    WHERE total >= 3
    WITH u, t, total, ac_count, (total - ac_count) * 1.0 / total AS error_rate
    WHERE error_rate > 0.3
    ORDER BY error_rate DESC, total DESC
    LIMIT 3
    MATCH (t)<-[:BELONGS_TO]-(rec:Problem)
    WHERE NOT EXISTS {
        MATCH (u)-[:SUBMITTED]->(sub:Submission)-[:FOR]->(rec)
        WHERE sub.result = '0'
    }
    RETURN DISTINCT rec.problem_id AS id, rec._id AS display_id, rec.title AS title,
           t.name AS weak_topic, rec.difficulty AS difficulty,
           rec.accepted_number AS ac_num
    ORDER BY ac_num DESC
    LIMIT 10
    """
    try:
        result = client.run_query(weak_query, {'username': username})
        for r in result:
            add_rec(r['id'], r.get('display_id', ''), r.get('title', ''), f"巩固薄弱知识点「{r['weak_topic']}」", score=85)
    except Exception as e:
        logger.error(f"薄弱知识点巩固查询失败: {e}")

    strength_query = """
    MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)-[:BELONGS_TO]->(t:Topic)
    WHERE s.result = '0'
    WITH u, t, p, count(s) AS ac_count
    WHERE ac_count >= 1
    WITH u, t, collect(DISTINCT p.difficulty) AS difficulties, ac_count
    ORDER BY ac_count DESC
    LIMIT 3
    UNWIND difficulties AS diff
    MATCH (t)<-[:BELONGS_TO]-(rec:Problem)
    WHERE rec.difficulty IN difficulties
    AND NOT EXISTS { MATCH (u)-[:SUBMITTED]->(:Submission)-[:FOR]->(rec) }
    RETURN DISTINCT rec.problem_id AS id, rec._id AS display_id, rec.title AS title,
           t.name AS strength_topic, rec.difficulty AS difficulty,
           rec.accepted_number AS ac_num
    ORDER BY ac_num DESC
    LIMIT 10
    """
    try:
        result = client.run_query(strength_query, {'username': username})
        for r in result:
            add_rec(r['id'], r.get('display_id', ''), r.get('title', ''), f"拓展擅长知识点「{r['strength_topic']}」的同难度题目", score=75)
    except Exception as e:
        logger.error(f"擅长知识点拓展查询失败: {e}")

    progression_query = """
    MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)
    WHERE s.result = '0'
    WITH u, collect(DISTINCT p.difficulty) AS user_difficulties
    UNWIND user_difficulties AS diff
    WITH u, diff
    ORDER BY diff
    WITH u, collect(diff)[-1] AS max_diff
    MATCH (rec:Problem)
    WHERE NOT EXISTS { MATCH (u)-[:SUBMITTED]->(:Submission)-[:FOR]->(rec) }
    AND (
        (max_diff = 'Low' AND rec.difficulty IN ['Low', 'Mid']) OR
        (max_diff = 'Mid' AND rec.difficulty IN ['Mid', 'High'])
    )
    RETURN rec.problem_id AS id, rec._id AS display_id, rec.title AS title,
           rec.difficulty AS difficulty, rec.accepted_number AS ac_num
    ORDER BY ac_num DESC
    LIMIT 10
    """
    try:
        result = client.run_query(progression_query, {'username': username})
        for r in result:
            add_rec(r['id'], r.get('display_id', ''), r.get('title', ''), f"挑战更高难度「{r['difficulty']}」的题目", score=60)
    except Exception as e:
        logger.error(f"难度递进推荐查询失败: {e}")

    recs.sort(key=lambda x: x['score'], reverse=True)
    return recs[:limit]


def _get_cf_recommendations(username: str, limit: int = 30) -> List[Dict]:
    client = neo4j_client
    recs = []
    seen_ids = set()

    cf_query = """
    MATCH (u:User {username: $username})-[:SUBMITTED]->(s1:Submission)-[:FOR]->(p:Problem)
    WHERE s1.result = '0'
    WITH u, collect(DISTINCT p) AS u_ac
    MATCH (other:User)-[:SUBMITTED]->(s2:Submission)-[:FOR]->(p)
    WHERE s2.result = '0' AND other <> u AND p IN u_ac
    WITH u, u_ac, other, count(DISTINCT p) AS common
    WHERE common >= 2
    ORDER BY common DESC
    LIMIT 5
    MATCH (other)-[:SUBMITTED]->(s3:Submission)-[:FOR]->(rec:Problem)
    WHERE s3.result = '0' AND NOT rec IN u_ac
    RETURN DISTINCT rec.problem_id AS id, rec._id AS display_id, rec.title AS title,
           rec.accepted_number AS ac_num
    ORDER BY ac_num DESC
    LIMIT $limit
    """
    try:
        result = client.run_query(cf_query, {'username': username, 'limit': limit})
        for r in result:
            pid = r['id']
            if pid not in seen_ids:
                seen_ids.add(pid)
                recs.append({'id': r.get('display_id', pid), '_id': r.get('display_id', pid), 'title': r.get('title', ''), 'reason': "与您学习路径相似的用户也做了此题", 'score': 65})
    except Exception as e:
        logger.error(f"协同过滤查询失败: {e}")

    return recs


def _get_gnn_recommendations(user_id: int, limit: int = 30) -> List[Dict]:
    try:
        engine = _get_fusion_engine()
        if engine is None or engine.gnn_embeddings is None:
            return []
        from recommend.gnn_recall import gnn_recall
        results = gnn_recall(user_id, engine.gnn_embeddings, engine.gnn_graph_data, top_k=limit)
        return [
            {'id': pid, 'reason': "基于知识图谱图神经网络推荐", 'score': max(score * 100, 55)}
            for pid, score in results
        ]
    except Exception as e:
        logger.warning(f"GNN召回失败: {e}")
        return []


def _get_sequence_recommendations(user_id: int, limit: int = 30) -> List[Dict]:
    try:
        engine = _get_fusion_engine()
        if engine is None or engine.sequence_model is None:
            return []
        from recommend.sequence_recall import sequence_recall
        results = sequence_recall(user_id, engine.sequence_model, engine.sequence_data, top_k=limit)
        return [
            {'id': pid, 'reason': "基于您的做题序列智能推荐", 'score': max(score * 100, 55)}
            for pid, score in results
        ]
    except Exception as e:
        logger.warning(f"序列召回失败: {e}")
        return []


def _apply_deepfm_rerank(user_id: int, candidates: List[Dict]) -> Optional[List[Dict]]:
    try:
        engine = _get_fusion_engine()
        if engine is None or engine.ranking_model is None:
            return None

        from recommend.deepfm_ranking import deepfm_rank
        candidate_ids = [c['id'] for c in candidates]
        ranked = deepfm_rank(user_id, candidate_ids, engine.ranking_model, engine.ranking_data)

        scored_map = dict(ranked)
        for c in candidates:
            c['score'] = scored_map.get(c['id'], c['score'])

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    except Exception as e:
        logger.warning(f"DeepFM重排失败，使用召回分数: {e}")
        return None


def _get_hot_recommendations(user, limit: int = 20) -> List[Dict]:
    from submission.models import Submission, JudgeStatus
    from problem.models import ProblemTag, Problem

    done_ids = Submission.objects.filter(user_id=user.id).values_list('problem_id', flat=True).distinct()
    problems = Problem.objects.exclude(id__in=done_ids).order_by('-accepted_number')[:limit]
    recs = []
    for p in problems:
        user_tags = ProblemTag.objects.filter(
            problem__submission__user_id=user.id,
            problem__submission__result=JudgeStatus.ACCEPTED
        ).distinct()
        common_tags = list(p.tags.filter(id__in=user_tags).values_list('name', flat=True))
        if common_tags:
            reason = f"基于您常做的「{common_tags[0]}」题目推荐"
        else:
            first_tag = p.tags.first()
            if first_tag:
                reason = f"热门「{first_tag.name}」题目推荐"
            else:
                reason = "热门题目推荐"
        recs.append({
            'id': p.id,
            'reason': reason
        })
    return recs


class RecommendAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='RecommendAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get('user_id')
        existing_profile = context.get('existing_profile', {})
        user_message = context.get('message', '')
        limit = context.get('limit', 10)
        offset = context.get('offset', 0)

        if not user_id:
            return {
                'success': False,
                'error': '请先登录',
                'agent': 'RecommendAgent',
            }

        from account.models import User
        user = User.objects.filter(id=user_id).first()
        if not user:
            return {
                'success': False,
                'error': '用户不存在',
                'agent': 'RecommendAgent',
            }

        username = user.username

        logger.info(f"RecommendAgent: user_id={user_id}, limit={limit}")

        fusion_recs = []
        seen_ids = set()

        def add_rec(pid, reason, score, source):
            if pid not in seen_ids:
                seen_ids.add(pid)
                fusion_recs.append({'id': pid, 'reason': reason, 'score': score, 'source': source})

        graph_recs = _get_graph_recommendations(username, limit=50)
        logger.info(f"  [规则图谱召回] {len(graph_recs)} 条")
        for rec in graph_recs:
            add_rec(rec['id'], rec['reason'], rec['score'], 'graph_rule')

        cf_recs = _get_cf_recommendations(username, limit=30)
        logger.info(f"  [协同过滤召回] {len(cf_recs)} 条")
        for rec in cf_recs:
            add_rec(rec['id'], rec['reason'], rec['score'], 'graph_cf')

        gnn_recs = _get_gnn_recommendations(user_id, limit=30)
        logger.info(f"  [图神经网络召回] {len(gnn_recs)} 条")
        for rec in gnn_recs:
            add_rec(rec['id'], rec['reason'], rec['score'], 'gnn')

        seq_recs = _get_sequence_recommendations(user_id, limit=30)
        logger.info(f"  [序列行为召回] {len(seq_recs)} 条")
        for rec in seq_recs:
            add_rec(rec['id'], rec['reason'], rec['score'], 'sequence')

        total_before_hot = len(fusion_recs)
        if total_before_hot < limit + offset:
            logger.info("召回不足，使用热度兜底")
            hot_recs = _get_hot_recommendations(user, limit=50)
            logger.info(f"  [热度兜底] {len(hot_recs)} 条")
            for rec in hot_recs:
                add_rec(rec['id'], rec['reason'], rec.get('score', 30), 'hot')

        logger.info(f"  [去重后候选总数] {len(fusion_recs)} 条")

        deepfm_ranked = _apply_deepfm_rerank(user_id, fusion_recs)
        if deepfm_ranked is not None:
            fusion_recs = deepfm_ranked
            logger.info(f"  [DeepFM精排] 完成排序")
        else:
            fusion_recs.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"  [Fallback排序] 使用召回分数排序")

        paged_recs = fusion_recs[offset:offset + limit]
        total = len(fusion_recs)

        from problem.models import Problem
        problem_ids = [rec['id'] for rec in paged_recs]
        problems_map = {
            p.id: p for p in Problem.objects.filter(id__in=problem_ids).prefetch_related('tags')
        }

        recommendations = []
        for rec in paged_recs:
            problem = problems_map.get(rec['id'])
            if not problem:
                continue
            source_tag = f" [{rec.get('source', '')}]" if rec.get('source') and rec['source'] != 'fusion' else ""
            recommendations.append({
                '_id': problem._id,
                'id': problem.id,
                'title': problem.title,
                'difficulty': problem.difficulty,
                'tags': [tag.name for tag in problem.tags.all()],
                'description': problem.description,
                'input_description': problem.input_description,
                'output_description': problem.output_description,
                'samples': problem.samples,
                'time_limit': problem.time_limit,
                'memory_limit': problem.memory_limit,
                'accepted_number': problem.accepted_number,
                'submission_number': problem.submission_number,
                'reason': rec['reason'] + source_tag,
                'score': round(rec['score'], 4),
                'source': rec.get('source', ''),
            })

        logger.info(f"最终返回推荐数量: {len(recommendations)}")

        return {
            'success': True,
            'recommendations': recommendations,
            'total': total,
            'total_before_hot': total_before_hot,
            'agent': 'RecommendAgent',
            'intent': 'recommend',
            'message': f'为您推荐了 {len(recommendations)} 道题目',
        }
