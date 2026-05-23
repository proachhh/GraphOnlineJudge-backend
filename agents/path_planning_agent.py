import logging
from typing import Any, Dict, List, Optional
from statistics import pstdev

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

MAX_PATH_DEPTH = 10
FALLBACK_DEPTH = 3


def _safe_float(val, default: float = 3.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _parse_goal_from_goals(onboarding_answers: Dict[str, str]) -> Optional[str]:
    goals_text = onboarding_answers.get('learning_goals', '')
    if not goals_text:
        return None

    from utils.neo4j_client import neo4j_client as client
    result = client.run_query("MATCH (t:Topic) RETURN t.name AS name")
    all_topics = {r['name'] for r in result}

    for topic in all_topics:
        if topic in goals_text:
            return topic

    return None


def _get_default_goal() -> Optional[str]:
    try:
        result = neo4j_client.run_query("""
            MATCH (:Topic)-[r:PREREQUISITE_OF]->(t:Topic)
            WITH t, count(r) AS incoming
            ORDER BY incoming DESC
            LIMIT 1
            RETURN t.name AS name
        """)
        if result:
            return result[0].get('name')
    except Exception as e:
        logger.warning(f"Failed to find default goal: {e}")
    return None


def _find_shortest_paths(start: str, goal: str) -> List[List[Dict]]:
    try:
        result = neo4j_client.run_query("""
            MATCH (start:Topic {name: $start}), (goal:Topic {name: $goal})
            MATCH path = shortestPath((start)-[:PREREQUISITE_OF*1..10]->(goal))
            RETURN [node in nodes(path) | {
                name: node.name,
                difficulty: coalesce(toFloat(node.difficulty), 3.0),
                importance: coalesce(toInteger(node.importance), 3)
            }] AS nodes_list
        """, {'start': start, 'goal': goal, 'max_depth': MAX_PATH_DEPTH})
        paths = []
        for row in result:
            paths.append(row['nodes_list'])
        return paths
    except Exception as e:
        logger.warning(f"No shortest path found from '{start}' to '{goal}': {e}")
        return []


def _fallback_paths(start: str) -> List[List[Dict]]:
    try:
        result = neo4j_client.run_query("""
            MATCH (start:Topic {name: $start})
            MATCH path = (start)-[:PREREQUISITE_OF*1..3]->(reachable:Topic)
            RETURN DISTINCT [node in nodes(path) | {
                name: node.name,
                difficulty: coalesce(toFloat(node.difficulty), 3.0),
                importance: coalesce(toInteger(node.importance), 3)
            }] AS nodes_list
            LIMIT 6
        """, {'start': start})
        paths = []
        for row in result:
            if row.get('nodes_list'):
                paths.append(row['nodes_list'])
        return paths
    except Exception as e:
        logger.warning(f"Fallback query failed: {e}")
        return []


def _get_problem_count(topic_name: str) -> int:
    try:
        result = neo4j_client.run_query("""
            MATCH (t:Topic {name: $name})<-[r:BELONGS_TO]-(p:Problem)
            RETURN count(r) AS cnt
        """, {'name': topic_name})
        if result:
            return result[0].get('cnt', 0)
    except Exception:
        pass
    return 0


def _search_rag_snippets(topic_name: str) -> str:
    try:
        from utils.vector_store import get_vector_store
        store = get_vector_store('oj_documents')
        if not store.is_ready:
            return ''
        results = store.search(topic_name, top_k=2)
        if not results:
            return ''
        snippets = [r.get('content', '')[:200] for r in results if r.get('content')]
        return ' '.join(snippets)
    except Exception as e:
        logger.warning(f"RAG search failed for '{topic_name}': {e}")
        return ''


class PathPlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='PathPlanningAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get('user_id')
        existing_profile = context.get('existing_profile', {})
        user_message = context.get('message', '')
        manual_start = context.get('start_topic', '')
        manual_goal = context.get('target_topic', '')

        if not existing_profile:
            try:
                from agents.master_agent import master_agent
                existing_profile = master_agent.load_user_profile(user_id)
            except Exception:
                pass

        weak_topics = existing_profile.get('weak_topics', [])

        if manual_start:
            start = manual_start
        elif weak_topics:
            start = weak_topics[0]
        else:
            return {
                'success': False,
                'error': '尚未收集到薄弱知识点信息，请先完成学习画像引导对话',
                'agent': 'PathPlanningAgent',
            }

        if manual_goal:
            goal = manual_goal
        else:
            onboarding_answers = existing_profile.get('_onboarding_answers', {})
            goal = _parse_goal_from_goals(onboarding_answers)
            if not goal:
                goal = _get_default_goal()

        if not goal:
            return {
                'success': False,
                'error': f'未找到可用的目标知识点。起点: {start}',
                'agent': 'PathPlanningAgent',
            }

        logger.info(f"PathPlanning: start={start}, goal={goal}")

        paths = _find_shortest_paths(start, goal)

        use_fallback = False
        if not paths:
            paths = _fallback_paths(start)
            use_fallback = True

        if not paths:
            return {
                'success': False,
                'error': f'从「{start}」出发深度 {FALLBACK_DEPTH} 内未发现可达的知识点',
                'agent': 'PathPlanningAgent',
            }

        best_path = self._select_best_path(paths)

        enriched = []
        for node in best_path:
            name = node.get('name', '')
            enriched.append({
                'name': name,
                'difficulty': _safe_float(node.get('difficulty'), 3.0),
                'importance': _safe_float(node.get('importance'), 3.0),
                'problem_count': _get_problem_count(name),
                'snippet': _search_rag_snippets(name),
            })

        return {
            'success': True,
            'path_plan': enriched,
            'path_count': len(enriched),
            'start': start,
            'goal': goal,
            'fallback': use_fallback,
            'agent': 'PathPlanningAgent',
            'intent': 'learning_path',
        }

    def _select_best_path(self, paths: List[List[Dict]]) -> List[Dict]:
        if len(paths) == 1:
            return paths[0]

        best_path = paths[0]
        best_stddev = float('inf')

        for path in paths:
            if len(path) < 2:
                continue
            difficulties = [_safe_float(n.get('difficulty'), 3.0) for n in path]
            try:
                std = pstdev(difficulties)
            except Exception:
                std = float('inf')
            if std < best_stddev:
                best_stddev = std
                best_path = path

        return best_path
