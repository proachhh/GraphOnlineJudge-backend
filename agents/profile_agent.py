import json
import logging
from typing import Any, Dict, List, Optional
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

ONBOARDING_DIMENSIONS = [
    'background',
    'current_courses',
    'weak_areas',
    'learning_goals',
    'learning_style',
    'weekly_hours',
]

ONBOARDING_QUESTIONS = {
    'background': {
        'question': '请问你的专业背景是什么？（例如：计算机科学、软件工程、数学等）',
        'dimension': 'background',
        'display': '专业背景',
    },
    'current_courses': {
        'question': '你目前正在学习哪些课程或知识点？（例如：数据结构、算法设计、操作系统等）',
        'dimension': 'current_courses',
        'display': '当前学习课程',
    },
    'weak_areas': {
        'question': '你觉得自己在哪些知识点或技能上比较薄弱，希望重点提升？（例如：动态规划、图论、递归等）',
        'dimension': 'weak_areas',
        'display': '知识薄弱点',
    },
    'learning_goals': {
        'question': '你的学习目标是什么？可以分别说说短期目标（近1-3个月）和长期目标（半年到一年）。',
        'dimension': 'learning_goals',
        'display': '学习目标',
    },
    'learning_style': {
        'question': '你更偏好哪种学习方式？（例如：观看视频教程、阅读文档书籍、动手实操写代码、或多种方式结合）',
        'dimension': 'learning_style',
        'display': '偏好学习方式',
    },
    'weekly_hours': {
        'question': '你每周大约有多少时间可以用来学习编程或刷题？（例如：5小时以下、5-10小时、10-20小时、20小时以上）',
        'dimension': 'weekly_hours',
        'display': '每周学习时长',
    },
}

ONBOARDING_DIMENSION_ORDER = [
    'background', 'current_courses', 'weak_areas',
    'learning_goals', 'learning_style', 'weekly_hours',
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
        submission_event = context.get('submission_event')

        if submission_event:
            updated = self._process_submission_event(
                user_id, existing_profile, submission_event
            )
            if updated:
                self._persist_profile(user_id, updated)
            return {
                'success': True,
                'profile': updated or existing_profile,
                'updated_from': 'submission_event',
            }

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

    def start_onboarding(self, user_id: int) -> Dict[str, Any]:
        profile = self._load_profile_from_neo4j(user_id)

        answered = profile.get('_onboarding_answers', {})

        remaining = [
            dim for dim in ONBOARDING_DIMENSION_ORDER
            if dim not in answered
        ]

        if not remaining:
            profile.pop('_onboarding_answers', None)
            profile.pop('_onboarding_current_dim', None)
            profile['_onboarding_complete'] = True
            profile.pop('_updated_at', None)
            profile['_updated_at'] = datetime.now().isoformat()
            self._persist_profile(user_id, profile)
            return {
                'onboarding_complete': True,
                'profile': profile,
                'message': '引导对话已完成！现在已为你建立学习画像，可以开始使用各项功能了。',
            }

        current_dim = remaining[0]
        profile['_onboarding_current_dim'] = current_dim
        self._persist_profile(user_id, profile)

        question_info = ONBOARDING_QUESTIONS[current_dim]

        return {
            'onboarding_complete': False,
            'current_dimension': current_dim,
            'dimension_display': question_info['display'],
            'question': question_info['question'],
            'step': len(answered) + 1,
            'total_steps': len(ONBOARDING_DIMENSION_ORDER),
            'remaining_dimensions': remaining,
        }

    def process_onboarding_answer(self, user_id: int,
                                   answer: str) -> Dict[str, Any]:
        profile = self._load_profile_from_neo4j(user_id)

        current_dim = profile.get('_onboarding_current_dim')
        if not current_dim:
            return self.start_onboarding(user_id)

        answered = profile.get('_onboarding_answers', {})
        answered[current_dim] = answer
        profile['_onboarding_answers'] = answered
        profile.pop('_onboarding_current_dim', None)
        profile['_updated_at'] = datetime.now().isoformat()

        self._persist_profile(user_id, profile)

        remaining = [
            dim for dim in ONBOARDING_DIMENSION_ORDER
            if dim not in answered
        ]

        if remaining:
            next_dim = remaining[0]
            profile['_onboarding_current_dim'] = next_dim
            self._persist_profile(user_id, profile)

            question_info = ONBOARDING_QUESTIONS[next_dim]
            return {
                'onboarding_complete': False,
                'current_dimension': next_dim,
                'dimension_display': question_info['display'],
                'question': question_info['question'],
                'step': len(answered) + 1,
                'total_steps': len(ONBOARDING_DIMENSION_ORDER),
                'previous_answer_stored': current_dim,
            }

        profile.pop('_onboarding_answers', None)
        profile.pop('_onboarding_current_dim', None)
        profile['_onboarding_complete'] = True

        answers_text = '；'.join([
            f"{ONBOARDING_QUESTIONS[k]['display']}: {v}"
            for k, v in answered.items()
        ])

        clean_existing = {
            k: v for k, v in profile.items()
            if not k.startswith('_')
        }

        result = self.run({
            'user_id': user_id,
            'message': answers_text,
            'existing_profile': clean_existing,
        })

        profile = result.get('profile', profile)
        profile['_onboarding_complete'] = True
        profile['_updated_at'] = datetime.now().isoformat()
        self._persist_profile(user_id, profile)

        return {
            'onboarding_complete': True,
            'profile': profile,
            'message': '引导对话已完成！现在已为你建立学习画像，可以开始使用各项功能了。',
        }

    def _process_submission_event(self, user_id: int,
                                   existing_profile: Dict[str, Any],
                                   event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = event.get('result')
        tags = event.get('tags', [])
        problem_title = event.get('problem_title', '')

        if not tags:
            return None

        profile = dict(existing_profile) if existing_profile else {}

        from submission.models import JudgeStatus

        for tag in tags:
            if result == JudgeStatus.ACCEPTED:
                if tag in profile.get('weak_topics', []):
                    weaks = [w for w in profile.get('weak_topics', []) if w != tag]
                    profile['weak_topics'] = weaks

                strengths = list(set(profile.get('strength_topics', []) + [tag]))
                profile['strength_topics'] = strengths[:8]
            else:
                if tag not in profile.get('weak_topics', []):
                    weaks = list(set(profile.get('weak_topics', []) + [tag]))
                    profile['weak_topics'] = weaks[:8]

                if tag in profile.get('strength_topics', []):
                    strengths = [s for s in profile.get('strength_topics', []) if s != tag]
                    profile['strength_topics'] = strengths

        profile['_updated_at'] = datetime.now().isoformat()
        return profile

    def _fetch_user_statistics(self, user_id: int) -> str:
        try:
            from submission.models import Submission, JudgeStatus
            from problem.models import ProblemTag
            from django.db.models import Count, Q

            total = Submission.objects.filter(user_id=user_id).count()
            ac_count = Submission.objects.filter(
                user_id=user_id, result=JudgeStatus.ACCEPTED
            ).count()
            accuracy = round(ac_count / total * 100, 1) if total > 0 else 0

            tag_stats = ProblemTag.objects.filter(
                problem__submission__user_id=user_id
            ).annotate(
                total=Count('problem__submission',
                            filter=Q(problem__submission__user_id=user_id)),
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
                f"知识点统计:\n" + '\n'.join(tag_lines)
                if tag_lines else "知识点统计: 暂无足够数据"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch user stats for {user_id}: {e}")
            return "暂无统计数据"

    def _build_prompt(self, user_message: str, stats: str,
                      existing_profile: Dict[str, Any]) -> str:
        existing_json = json.dumps(
            existing_profile, ensure_ascii=False, indent=2
        ) if existing_profile else '暂无'

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
                profile = json.loads(
                    '{' + raw_response.split('{', 1)[1].rsplit('}', 1)[0] + '}'
                )

            result = {}
            for dim in PROFILE_DIMENSIONS:
                default = [] if dim in ('strength_topics', 'weak_topics') else ''
                result[dim] = profile.get(dim, default)
            return result
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(
                f"Failed to parse profile JSON: {e}, "
                f"raw='{raw_response[:200]}...'"
            )
            return {
                dim: (
                    [] if dim in ('strength_topics', 'weak_topics') else ''
                )
                for dim in PROFILE_DIMENSIONS
            }

    def _merge_with_existing(self, new_profile: Dict[str, Any],
                              existing: Dict[str, Any]) -> Dict[str, Any]:
        if not existing:
            return new_profile

        merged = dict(existing)

        for dim in ['_onboarding_answers', '_onboarding_current_dim',
                     '_onboarding_complete']:
            if dim in merged:
                new_profile[dim] = merged[dim]

        for dim in PROFILE_DIMENSIONS:
            if dim in new_profile and new_profile[dim]:
                merged[dim] = new_profile[dim]

        merged['_updated_at'] = datetime.now().isoformat()
        return merged

    def _load_profile_from_neo4j(self, user_id: int) -> Dict[str, Any]:
        try:
            result = neo4j_client.run_query(
                """
                MATCH (u:User {user_id: $user_id})
                RETURN u.profile_knowledge_mastery AS mastery,
                       u.profile_strength_topics AS strengths,
                       u.profile_weak_topics AS weaks,
                       u.profile_coding_style AS style,
                       u.profile_learning_pace AS pace,
                       u.profile_recommended_focus AS focus,
                       u.profile_onboarding_answers AS onboarding_answers,
                       u.profile_onboarding_current_dim AS current_dim,
                       u.profile_onboarding_complete AS complete,
                       u.profile_updated_at AS updated
                """,
                {'user_id': user_id}
            )
            if result and len(result) > 0:
                r = result[0]
                onboarding_answers = r.get('onboarding_answers')
                if isinstance(onboarding_answers, str):
                    try:
                        onboarding_answers = json.loads(onboarding_answers)
                    except json.JSONDecodeError:
                        onboarding_answers = {}

                return {
                    'knowledge_mastery': r.get('mastery', ''),
                    'strength_topics': r.get('strengths') or [],
                    'weak_topics': r.get('weaks') or [],
                    'coding_style': r.get('style', ''),
                    'learning_pace': r.get('pace', ''),
                    'recommended_focus': r.get('focus', ''),
                    '_onboarding_answers': onboarding_answers or {},
                    '_onboarding_current_dim': r.get('current_dim', ''),
                    '_onboarding_complete': r.get('complete', False),
                    '_updated_at': r.get('updated', ''),
                }
        except Exception as e:
            logger.warning(f"Failed to load profile from Neo4j: {e}")

        return {
            'knowledge_mastery': '',
            'strength_topics': [],
            'weak_topics': [],
            'coding_style': '',
            'learning_pace': '',
            'recommended_focus': '',
            '_onboarding_answers': {},
            '_onboarding_current_dim': '',
            '_onboarding_complete': False,
            '_updated_at': '',
        }

    def get_clean_profile(self, user_id: int) -> Dict[str, Any]:
        profile = self._load_profile_from_neo4j(user_id)
        for key in list(profile.keys()):
            if key.startswith('_'):
                del profile[key]
        return profile

    def _persist_profile(self, user_id: int, profile: Dict[str, Any]):
        try:
            onboarding_answers = profile.get('_onboarding_answers', {})
            if isinstance(onboarding_answers, dict):
                onboarding_answers = json.dumps(onboarding_answers, ensure_ascii=False)

            updated_str = profile.get('_updated_at', '')
            if not updated_str:
                updated_str = datetime.now().isoformat()

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
                    u.profile_onboarding_answers = $onboarding_answers,
                    u.profile_onboarding_current_dim = $current_dim,
                    u.profile_onboarding_complete = $complete,
                    u.profile_updated_at = CASE WHEN $updated = '' OR $updated IS NULL THEN datetime() ELSE datetime($updated) END
                """,
                {
                    'user_id': user_id,
                    'mastery': profile.get('knowledge_mastery', ''),
                    'strengths': profile.get('strength_topics', []),
                    'weaks': profile.get('weak_topics', []),
                    'style': profile.get('coding_style', ''),
                    'pace': profile.get('learning_pace', ''),
                    'focus': profile.get('recommended_focus', ''),
                    'onboarding_answers': onboarding_answers or '',
                    'current_dim': profile.get('_onboarding_current_dim', ''),
                    'complete': profile.get('_onboarding_complete', False),
                    'updated': updated_str,
                }
            )
            logger.info(f"Profile persisted to Neo4j for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to persist profile to Neo4j for user {user_id}: {e}")
