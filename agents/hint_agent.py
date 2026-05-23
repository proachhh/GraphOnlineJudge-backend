import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

HINT_SYSTEM_PROMPT = """# 角色
你是一个专业的编程竞赛辅导教练。你的任务是给学生提供解题提示，而不是直接给出答案。

# 规则
1. **渐进式提示**：从泛到精，先给方向性提示，再给具体思路
2. **不直接给答案**：永远不要给出完整代码或直接答案
3. **联系知识点**：提示中要关联到相关的算法/数据结构知识点
4. **引导思考**：鼓励学生自己推导，用提问的方式引导

# 输出格式
请按以下结构输出：

## 思路方向
（1-2句话，总体方向性提示，不要涉及具体实现）

## 关键知识点
列出本题涉及的核心知识点（2-4个）

## 分步提示
1. 第一步提示（最模糊）
2. 第二步提示（稍微具体）
3. 第三步提示（给出关键思路，但不给代码）

## 常见陷阱
（1-2个容易出错的地方，但不给出解决方案）"""


class HintAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='HintAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get('user_id')
        user_message = context.get('message', '')
        existing_profile = context.get('existing_profile', {})

        if not user_message:
            return {
                'success': False,
                'error': '请提供需要提示的题目信息',
                'agent': 'HintAgent',
            }

        topic = self._extract_topic(user_message)

        problem_info = self._find_related_problem(user_message)
        rag_context = self._rag_search(topic, user_message)
        kg_context = self._fetch_kg_context(topic)

        prompt = HINT_SYSTEM_PROMPT + f"""

# 学生的问题
{user_message}

# 参考资料（RAG检索结果）
{rag_context}

# 知识图谱上下文
{kg_context}

# 题目信息
{problem_info}

# 学生画像
强项: {', '.join(existing_profile.get('strength_topics', []) or ['暂无'])}
弱项: {', '.join(existing_profile.get('weak_topics', []) or ['暂无'])}

请基于以上信息为学生生成解题提示。"""

        try:
            from aiChat.utils import ask_ai
            response = ask_ai(prompt)
        except Exception as e:
            logger.error(f"AI call failed: {e}")
            return {
                'success': False,
                'error': f'AI 服务调用失败: {str(e)}',
                'agent': 'HintAgent',
            }

        return {
            'success': True,
            'hint': response,
            'topic': topic,
            'display_type': 'hint',
            'agent': 'HintAgent',
            'intent': 'hint',
        }

    def _extract_topic(self, message: str) -> str:
        from utils.neo4j_client import neo4j_client as client
        result = client.run_query("MATCH (t:Topic) RETURN t.name AS name")
        all_topics = {r['name'] for r in (result or [])}

        for topic in all_topics:
            if topic in message:
                return topic
        return ''

    def _find_related_problem(self, message: str) -> str:
        try:
            from problem.models import Problem

            for word in message.split():
                if len(word) >= 3:
                    problems = Problem.objects.filter(title__icontains=word, visible=True)[:3]
                    if problems:
                        lines = []
                        for p in problems:
                            lines.append(f"- {p._id}: {p.title} (难度: {p.difficulty})")
                        return '\n'.join(lines)
            return '（未找到关联题目）'
        except Exception:
            return '（未找到关联题目）'

    def _rag_search(self, topic: str, query: str) -> str:
        try:
            from utils.vector_store import get_vector_store
            store = get_vector_store('oj_documents')
            if not store.is_ready:
                return '（暂无参考资料）'
            search_query = f"{topic} {query}"[:200]
            results = store.search(search_query, top_k=3)
            if not results:
                return '（暂无参考资料）'
            parts = []
            for i, doc in enumerate(results):
                content = doc.get('content', '')[:800]
                parts.append(f"【参考资料 {i+1}】\n{content}")
            return '\n---\n'.join(parts)
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return '（暂无参考资料）'

    def _fetch_kg_context(self, topic: str) -> str:
        if not topic:
            return '（未识别到具体知识点）'
        try:
            client = neo4j_client
            query = """
            MATCH (t:Topic {name: $topic})
            OPTIONAL MATCH (t)-[:PREREQUISITE_OF]->(next:Topic)
            OPTIONAL MATCH (prev:Topic)-[:PREREQUISITE_OF]->(t)
            OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Topic)
            RETURN t.name AS topic,
                   t.difficulty AS difficulty,
                   collect(DISTINCT prev.name) AS prerequisites,
                   collect(DISTINCT next.name) AS successors,
                   collect(DISTINCT related.name) AS related_topics
            """
            result = client.run_query(query, {'topic': topic})
            if not result:
                return f'目标知识点: {topic}'
            r = result[0]
            lines = [f"目标知识点: {r.get('topic', topic)}"]
            if r.get('difficulty') is not None:
                lines.append(f"难度: {r['difficulty']}")
            prereqs = [x for x in (r.get('prerequisites') or []) if x]
            if prereqs:
                lines.append(f"前置知识: {', '.join(prereqs)}")
            successors = [x for x in (r.get('successors') or []) if x]
            if successors:
                lines.append(f"后继知识: {', '.join(successors)}")
            return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"KG query failed: {e}")
            return f'目标知识点: {topic}'
