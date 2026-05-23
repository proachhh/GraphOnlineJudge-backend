import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

ERROR_ANALYSIS_PROMPT = """# 角色
你是一个专业的编程竞赛教练，擅长分析学生的提交错误并给出改进建议。

# 规则
1. 根据提交历史和错误类型，诊断学生的问题所在
2. 给出具体的改进建议，而不是泛泛而谈
3. 如果识别到知识点薄弱，推荐相应的巩固题目
4. 语言要鼓励性，不要打击学生信心

# 输出格式
请按以下结构输出：

## 错误诊断
（分析最近几次提交错误的共同特征和根本原因）

## 知识点薄弱环节
（列出可能薄弱的知识点，如果信息不足可以写"暂无法判断"）

## 改进建议
（3-5条具体可操作的建议，按优先级排列）

## 推荐练习
（建议练习的方向或题型，不具体到题目ID）"""


class ErrorAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='ErrorAnalysisAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get('user_id')
        user_message = context.get('message', '')
        existing_profile = context.get('existing_profile', {})

        if not user_id:
            return {
                'success': False,
                'error': '请先登录',
                'agent': 'ErrorAnalysisAgent',
            }

        submission_history = self._get_recent_errors(user_id)
        problem_summary = self._summarize_problems(submission_history)

        if not submission_history:
            return {
                'success': True,
                'analysis': '你最近没有失败的提交记录，继续保持！',
                'submission_history': [],
                'agent': 'ErrorAnalysisAgent',
                'intent': 'analyze_error',
            }

        rag_context = self._search_errors(user_message)
        kg_context = self._fetch_user_kg_context(user_id)

        prompt = ERROR_ANALYSIS_PROMPT + f"""

# 学生的问题
{user_message}

# 学生画像
强项: {', '.join(existing_profile.get('strength_topics', []) or ['暂无'])}
弱项: {', '.join(existing_profile.get('weak_topics', []) or ['暂无'])}

# 最近错误提交
{problem_summary}

# 知识图谱上下文
{kg_context}

# 参考资料
{rag_context}

请基于以上信息为学生做错误分析。"""

        try:
            from aiChat.utils import ask_ai
            response = ask_ai(prompt)
        except Exception as e:
            logger.error(f"AI call failed: {e}")
            return {
                'success': False,
                'error': f'AI 服务调用失败: {str(e)}',
                'agent': 'ErrorAnalysisAgent',
            }

        return {
            'success': True,
            'analysis': response,
            'submission_count': len(submission_history),
            'display_type': 'error_analysis',
            'agent': 'ErrorAnalysisAgent',
            'intent': 'analyze_error',
        }

    def _get_recent_errors(self, user_id: int, limit: int = 10) -> List[Dict]:
        from submission.models import Submission, JudgeStatus

        submissions = Submission.objects.filter(
            user_id=user_id
        ).exclude(
            result=JudgeStatus.ACCEPTED
        ).exclude(
            result=JudgeStatus.PENDING
        ).exclude(
            result=JudgeStatus.JUDGING
        ).order_by('-create_time')[:limit]

        result_map = {
            JudgeStatus.WRONG_ANSWER: '答案错误',
            JudgeStatus.COMPILE_ERROR: '编译错误',
            JudgeStatus.CPU_TIME_LIMIT_EXCEEDED: '时间超限',
            JudgeStatus.REAL_TIME_LIMIT_EXCEEDED: '时间超限',
            JudgeStatus.MEMORY_LIMIT_EXCEEDED: '内存超限',
            JudgeStatus.RUNTIME_ERROR: '运行错误',
            JudgeStatus.SYSTEM_ERROR: '系统错误',
            JudgeStatus.PARTIALLY_ACCEPTED: '部分正确',
        }

        results = []
        for s in submissions:
            result_text = result_map.get(s.result, f'错误码{s.result}')
            info = s.statistic_info or {}
            results.append({
                'problem_id': s.problem._id if s.problem_id else '',
                'problem_title': s.problem.title if s.problem_id else '',
                'result': result_text,
                'language': s.language,
                'time_cost': info.get('time_cost', ''),
                'memory_cost': info.get('memory_cost', ''),
                'err_info': info.get('err_info', ''),
                'create_time': s.create_time.isoformat() if s.create_time else '',
            })
        return results

    def _summarize_problems(self, submissions: List[Dict]) -> str:
        if not submissions:
            return '（暂无最近错误记录）'

        lines = [f'最近 {len(submissions)} 次错误提交：']
        for i, s in enumerate(submissions):
            detail = f"  题目: {s['problem_title']} ({s['problem_id']})"
            detail += f"  错误类型: {s['result']}"
            detail += f"  语言: {s['language']}"
            if s['time_cost']:
                detail += f"  耗时: {s['time_cost']}ms"
            if s['memory_cost']:
                detail += f"  内存: {s['memory_cost']}MB"
            if s['err_info']:
                detail += f"  错误信息: {s['err_info'][:200]}"
            lines.append(f"  {i+1}. {detail}")
        return '\n'.join(lines)

    def _search_errors(self, user_message: str) -> str:
        try:
            from utils.vector_store import get_vector_store
            store = get_vector_store('oj_documents')
            if not store.is_ready:
                return '（暂无参考资料）'
            results = store.search(user_message[:200], top_k=2)
            if not results:
                return '（暂无参考资料）'
            parts = []
            for i, doc in enumerate(results):
                content = doc.get('content', '')[:500]
                parts.append(f"【参考资料 {i+1}】\n{content}")
            return '\n---\n'.join(parts)
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return '（暂无参考资料）'

    def _fetch_user_kg_context(self, user_id: int) -> str:
        try:
            client = neo4j_client

            query = """
            MATCH (u:User {user_id: $user_id})-[:SUBMITTED]->(s:Submission)-[:FOR]->(:Problem)-[:BELONGS_TO]->(t:Topic)
            WHERE s.result <> '0'
            WITH t, count(s) AS error_count
            ORDER BY error_count DESC
            LIMIT 5
            RETURN t.name AS topic, error_count
            """
            result = client.run_query(query, {'user_id': user_id})
            if not result:
                return '（暂无知识图谱数据）'

            lines = ['错误集中在以下知识点：']
            for r in result:
                lines.append(f"  - {r['topic']} (错误 {r['error_count']} 次)")
            return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"KG query failed: {e}")
            return '（知识图谱查询失败）'
