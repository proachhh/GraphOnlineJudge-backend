import logging
from typing import Any, Dict, List, Optional
from statistics import pstdev

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

MAX_PATH_DEPTH = 10
FALLBACK_DEPTH = 3

# 中文Topic名 → 英文ProblemTag关键词映射
# 用于在Neo4j BELONGS_TO缺失时，回退到PostgreSQL查询
TOPIC_TO_TAG_KEYWORDS = {
    '数组': ['array', 'Array'],
    '栈': ['stack'],
    '队列': ['queue'],
    '串的基本概念': ['string', 'String', 'char', 'character'],
    '字符串匹配': ['string', 'String', 'stringMatch', 'kmp'],
    '二分查找': ['binarySearch', 'binary'],
    '二叉搜索树': ['binarySearch', 'binary'],
    '二叉树的基本概念': ['binary', 'BinaryTree', 'Binary Tree'],
    '二叉树的遍历方式': ['binary', 'dfs', 'bfs', 'BinaryTree', 'Binary Tree'],
    '优先队列': ['priorityQueue', 'heap', 'Heap'],
    '位运算技巧': ['bitmask', 'bit'],
    '冒泡排序': ['sorting', 'sort', 'sorted'],
    '分治算法': ['divideAndConquer', 'merge'],
    '前缀和': ['prefix', 'Prefix Sum', 'PrefixSum'],
    '动态规划': ['Dynamic Programming', 'DynamicProgramming', 'dp'],
    '单调栈': ['stack'],
    '单调队列': ['queue'],
    '单链表': ['linkedList', 'list'],
    '双向链表': ['linkedList', 'list'],
    '双指针': ['twoPointer'],
    '双端队列': ['queue', 'deque'],
    '哈夫曼树': ['binary', 'tree'],
    '哈希表-开放地址法': ['hash', 'Hash'],
    '哈希表-链地址法': ['hash', 'Hash'],
    '图的基本概念': ['graph', 'Graph'],
    '图的存储-邻接矩阵': ['graph', 'Graph', 'matrix', 'Matrix'],
    '图的存储-邻接表': ['graph', 'Graph'],
    '图的连通分量': ['graph', 'Graph', 'dfs'],
    '基数排序': ['radix', 'sorting', 'sort'],
    '堆排序': ['sorting', 'sort', 'heap', 'Heap'],
    '外部排序': ['sorting', 'sort'],
    '差分数组': ['prefix', 'array', 'Array'],
    '希尔排序': ['sorting', 'sort'],
    '平衡二叉树': ['binary', 'balanced', 'tree'],
    '并查集': ['unionFind', 'disjointSet', 'graph'],
    '广度优先搜索': ['bfs', 'BFS'],
    '强连通分量': ['graph', 'Graph', 'dfs'],
    '归并排序': ['merge', 'divideAndConquer', 'sorting', 'sort'],
    '循环链表': ['linkedList', 'list'],
    '循环队列': ['queue'],
    '快速排序': ['sorting', 'sort', 'quick'],
    '拓扑排序': ['topologicalSort', 'graph', 'Graph'],
    '插入排序': ['insertion', 'sorting', 'sort'],
    '时间复杂度': ['timeComplexity'],
    '最小生成树': ['graph', 'Graph', 'minimumSpanningTree', 'mst'],
    '最短路径': ['shortestPath', 'graph', 'Graph'],
    '朴素模式匹配': ['string', 'String', 'stringMatch'],
    '栈的应用-括号匹配': ['stack'],
    '栈的应用-表达式求值': ['stack'],
    '树': ['tree', 'binary', 'BinaryTree', 'Binary Tree'],
    '树状数组': ['fenwick', 'binaryIndexedTree'],
    '桶排序': ['sorting', 'sort', 'bucket'],
    '深度优先搜索': ['dfs', 'DFS'],
    '滑动窗口': ['sliding', 'slidingWindow', 'slidingPuzzle'],
    '空间复杂度': ['spaceComplexity'],
    '红黑树': ['binary', 'tree', 'balanced'],
    '线性表': ['array', 'Array', 'list'],
    '线段树': ['segmentTree'],
    '线索二叉树': ['binary', 'tree'],
    '计数排序': ['counting', 'countingSort', 'sorting', 'sort'],
    '记忆化搜索': ['dfs', 'memoization', 'dp'],
    '跳表': ['skipList', 'list'],
    '选择排序': ['selection', 'selectionSort', 'sorting', 'sort'],
    '递归': ['Recursion', 'recursion'],
    '链栈': ['stack', 'linkedList'],
    '链队列': ['queue', 'linkedList'],
    '顺序查找': ['search', 'simulation'],
    '顺序栈': ['stack', 'array', 'Array'],
    '顺序表': ['array', 'Array', 'list'],
    '0-1 背包问题': ['Dynamic Programming', 'DynamicProgramming', 'dp'],
    'AC 自动机': ['string', 'String', 'automaton'],
    'B 树': ['binary', 'tree'],
    'B+ 树': ['binary', 'tree'],
    'Boyer-Moore 算法': ['string', 'String', 'stringMatch'],
    'Floyd 判圈算法': ['graph', 'Graph', 'cycle'],
    'KMP 算法': ['string', 'String', 'stringMatch', 'kmp'],
    'SPFA 算法': ['graph', 'Graph', 'shortestPath'],
    'Trie 树': ['trie', 'string', 'String', 'tree'],
    '关键路径': ['graph', 'Graph', 'topologicalSort'],
    '指针与引用': ['pointer'],
    '迭代与遍历': ['iteration'],
    '逆波兰表达式': ['stack', 'expression'],
}



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


def _get_matching_tag_names(topic_name: str) -> list:
    """为 Neo4j Topic 名称在 PostgreSQL ProblemTag 中查找匹配的标签名"""
    if not topic_name:
        return []
    try:
        from problem.models import ProblemTag
        all_tags = list(ProblemTag.objects.values_list('name', flat=True).distinct())
    except Exception:
        return []

    # 1. 精确匹配
    if topic_name in all_tags:
        return [topic_name]

    topic_lower = topic_name.strip().lower()
    matches = set()

    # 2. 大小写不敏感精确匹配
    for tag in all_tags:
        if tag.strip().lower() == topic_lower:
            matches.add(tag)

    # 3. 包含匹配（tag 名包含 topic 名或反过来）
    if not matches:
        for tag in all_tags:
            tag_lower = tag.strip().lower()
            if topic_lower in tag_lower or tag_lower in topic_lower:
                if len(topic_lower) >= 3 or len(tag_lower) >= 3:
                    matches.add(tag)

    # 4. 单词级匹配（对于 "Dynamic Programming" ↔ "DynamicProgramming" 的情况）
    if not matches:
        topic_words = set(topic_lower.replace('-', ' ').replace('_', ' ').split())
        for tag in all_tags:
            tag_words = set(tag.strip().lower().replace('-', ' ').replace('_', ' ').split())
            if topic_words and tag_words:
                overlap = topic_words & tag_words
                if len(overlap) >= len(topic_words) * 0.5 or len(overlap) >= 2:
                    matches.add(tag)

    # 5. 中文关键词映射（解决中文Topic名与英文Tag名不匹配的问题）
    if not matches and topic_name in TOPIC_TO_TAG_KEYWORDS:
        keywords = TOPIC_TO_TAG_KEYWORDS[topic_name]
        for tag in all_tags:
            tag_lower_compact = tag.strip().lower().replace(' ', '').replace('-', '').replace('_', '')
            for kw in keywords:
                kw_compact = kw.lower().replace(' ', '').replace('-', '').replace('_', '')
                if tag_lower_compact == kw_compact:
                    matches.add(tag)
                    break
                if kw_compact in tag_lower_compact and len(kw_compact) >= 3:
                    matches.add(tag)
                    break

    return list(matches)[:5]


def _get_problem_count(topic_name: str) -> int:
    # 策略1：从 Neo4j BELONGS_TO 查询
    try:
        result = neo4j_client.run_query("""
            MATCH (t:Topic {name: $name})<-[r:BELONGS_TO]-(p:Problem)
            RETURN count(r) AS cnt
        """, {'name': topic_name})
        if result and result[0].get('cnt', 0) > 0:
            return result[0]['cnt']
    except Exception:
        pass

    # 策略2：通过 ProblemTag 名称多层匹配查 PostgreSQL
    try:
        from problem.models import Problem
        tag_names = _get_matching_tag_names(topic_name)
        if tag_names:
            count = Problem.objects.filter(
                tags__name__in=tag_names,
                visible=True,
                contest__isnull=True
            ).count()
            if count > 0:
                return count
    except Exception:
        pass

    return 0


def _get_representative_problems(topic_name: str, limit: int = 3) -> list:
    """获取某个知识点的代表性题目（标题+ID）"""
    try:
        from problem.models import Problem

        # 先尝试通过匹配的标签名查
        tag_names = _get_matching_tag_names(topic_name)
        qs = Problem.objects.filter(visible=True, contest__isnull=True)
        if tag_names:
            qs = qs.filter(tags__name__in=tag_names)
        else:
            # 没有匹配标签，返回空
            return []

        problems = qs.order_by('-submission_number')[:limit]
        return [{
            '_id': p._id,
            'id': p.id,
            'title': p.title,
            'difficulty': p.difficulty,
        } for p in problems]
    except Exception:
        return []


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
                'problems': _get_representative_problems(name, limit=3),
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
