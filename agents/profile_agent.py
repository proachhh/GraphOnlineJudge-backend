import json
import logging
from typing import Any, Dict
from datetime import datetime

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

PROFILE_DIMENSIONS = [
    'knowledge_mastery',
    'strength_topics',
    'weak_topics',
    'coding_style',
    'learning_pace',
    'recommended_focus',
]

PROFILE_SYSTEM_PROMPT = """# 角色
你是一个专业的编程学习分析专家。根据用户的做题数据和对话内容，精确抽取学习画像。

# 输出格式
你必须严格按照下面的 JSON 格式输出，不能包含任何其他文字：

```json
{
  "knowledge_mastery": "整体掌握度评价（1-2句话，包含强项和弱项）",
  "strength_topics": ["用户擅长的知识点1", "知识点2"],
  "weak_topics": ["用户薄弱的知识点1", "知识点2"],
  "coding_style": "编码风格描述（1-2句话，如：注重代码简洁性、常忽略边界条件等）",
  "learning_pace": "学习节奏评价（1-2句话，如：刷题频率高但深度不够）",
  "recommended_focus": "建议重点提升方向（1-2句话）"
}
```

# 规则
- 如果对话中明确描述了相关信息，优先使用对话内容
- 如果对话中信息不足，参考用户做题数据补充
- 如果所有信息都不足，用空字符串 "" 或空数组 [] 表示"""


class ProfileAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='ProfileAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context.get('user_id')
        user_message = context.get('message', '')
        existing_profile = context.get('existing_profile', {})

        user_stats_summary = self._fetch_user_statistics(user_id)

        prompt = self._build_prompt(user_message, user_stats_summary, existing_profile)

        try:
            from aiChat.utils import ask_ai
            raw_response = ask_ai(prompt)
        except Exception as e:
            logger.error(f"AI invocation failed: {e}")
            return {
                'success': False,
                'error': f'AI 服务调用失败: {str(e)}',
                'profile': existing_profile,
            }

        profile = self._parse_profile(raw_response)
        profile = self._merge_with_existing(profile, existing_profile)

        self._persist_profile(user_id, profile)

        return {
            'success': True,
            'profile': profile,
            'dimensions': list(profile.keys()),
        }

    def _fetch_user_statistics(self, user_id: int) -> str:
        try:
            from submission.models import Submission, JudgeStatus
            from problem.models import ProblemTag
            from django.db.models import Count, Q

            total = Submission.objects.filter(user_id=user_id).count()
            ac_count = Submission.objects.filter(user_id=user_id, result=JudgeStatus.ACCEPTED).count()
            accuracy = round(ac_count / total * 100, 1) if total > 0 else 0

            tag_stats = ProblemTag.objects.filter(
                problem__submission__user_id=user_id
            ).annotate(
                total=Count('problem__submission', filter=Q(problem__submission__user_id=user_id)),
                ac=Count('problem__submission',
                          filter=Q(problem__submission__user_id=user_id) &
                                  Q(problem__submission__result=JudgeStatus.ACCEPTED))
            ).filter(total__gt=2).order_by('-total')[:10]

            tag_lines = []
            for tag in tag_stats:
                rate = round(tag.ac / tag.total * 100, 1) if tag.total else 0
                tag_lines.append(f"  - {tag.name}: {tag.total}次提交, 正确率{rate}%")

            return (
                f"总提交数: {total}, AC数: {ac_count}, 整体正确率: {accuracy}%\n"
                f"知识点统计:\n" + '\n'.join(tag_lines) if tag_lines else "知识点统计: 暂无足够数据"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch user stats for {user_id}: {e}")
            return "暂无统计数据"

    def _build_prompt(self, user_message: str, stats: str,
                      existing_profile: Dict[str, Any]) -> str:
        existing_json = json.dumps(existing_profile, ensure_ascii=False, indent=2) if existing_profile else '暂无'

        return f"""{PROFILE_SYSTEM_PROMPT}

# 用户本次对话消息
{user_message if user_message else '无'}

# 用户做题统计数据
{stats}

# 用户已有学习画像（增量更新上下文）
{existing_json}

请基于以上所有信息，输出更新后的学习画像 JSON。"""
    def _parse_profile(self, raw_response: str) -> Dict[str, Any]:
        try:
            if '```json' in raw_response:
                start = raw_response.index('```json') + 7
                end = raw_response.index('```', start)
                raw_response = raw_response[start:end]
            elif '```' in raw_response:
                start = raw_response.index('```') + 3
                end = raw_response.index('```', start)
                raw_response = raw_response[start:end]

            raw_response = raw_response.strip()
            if raw_response.startswith('{'):
                profile = json.loads(raw_response)
            else:
                profile = json.loads('{' + raw_response.split('{', 1)[1].rsplit('}', 1)[0] + '}')

            result = {}
            for dim in PROFILE_DIMENSIONS:
                result[dim] = profile.get(dim, '' if dim != 'strength_topics' and dim != 'weak_topics' else [])
            return result
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"Failed to parse profile JSON: {e}, raw='{raw_response[:200]}...'")
            return {dim: '' if dim != 'strength_topics' and dim != 'weak_topics' else [] for dim in PROFILE_DIMENSIONS}

    def _merge_with_existing(self, new_profile: Dict[str, Any],
                              existing: Dict[str, Any]) -> Dict[str, Any]:
        if not existing:
            return new_profile
        for dim in PROFILE_DIMENSIONS:
            if dim in new_profile and new_profile[dim]:
                existing[dim] = new_profile[dim]
        existing['_updated_at'] = datetime.now().isoformat()
        return existing

    def _persist_profile(self, user_id: int, profile: Dict[str, Any]):
        try:
            client = neo4j_client
            client.run_query(
                """
                MERGE (u:User {user_id: $user_id})
                SET u.profile_knowledge_mastery = $mastery,
                    u.profile_strength_topics = $strengths,
                    u.profile_weak_topics = $weaks,
                    u.profile_coding_style = $style,
                    u.profile_learning_pace = $pace,
                    u.profile_recommended_focus = $focus,
                    u.profile_updated_at = datetime($updated)
                """,
                {
                    'user_id': user_id,
                    'mastery': profile.get('knowledge_mastery', ''),
                    'strengths': profile.get('strength_topics', []),
                    'weaks': profile.get('weak_topics', []),
                    'style': profile.get('coding_style', ''),
                    'pace': profile.get('learning_pace', ''),
                    'focus': profile.get('recommended_focus', ''),
                    'updated': profile.get('_updated_at', datetime.now().isoformat()),
                }
            )
            logger.info(f"Profile persisted to Neo4j for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to persist profile to Neo4j for user {user_id}: {e}")
