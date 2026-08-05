import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from agents.profile_agent import ProfileAgent
from agents.resource_agent import ResourceAgent
from agents.path_planning_agent import PathPlanningAgent
from agents.recommend_agent import RecommendAgent
from agents.hint_agent import HintAgent
from agents.error_analysis_agent import ErrorAnalysisAgent

logger = logging.getLogger(__name__)

INTENT_KEYWORDS = {
    'profile': ['画像', '分析我的学习', '我的水平', '我学得怎么样', '学习报告', '我的能力',
                '我的强项', '我的弱项', '知识掌握', '学习评估', '当前能力', '我的学习情况'],
    'recommend': ['推荐', '下一题', '做题建议', '刷题', '学什么', '推荐题目',
                  '有什么题目', '什么题适合我', '帮我推荐', '介绍几道', '来几道'],
    'hint': ['提示', '怎么做', '思路', '解题方向', '帮我理解', '这道题', '参考答案',
             '解答', '不会做', '教教我'],
    'analyze_error': ['为什么错了', '哪里错了', '错误分析', '帮我看看', '提交失败',
                      '运行错误', '编译错误', '超时', '为什么没过', '错在哪',
                      '提交错了', '哪里不对', '报错了', '为什么出错', '分析错误', '怎么错了'],
    'learning_path': ['学习路径', '学习路线', '怎么学', '学习计划', '进阶路线',
                      '先学什么', '路线图'],
    'resource': ['生成资料', '讲解', '出题', '思维导图', '阅读材料', '代码案例',
                 '生成讲稿', '整理笔记', '画图', '导图', '生成题目',
                 '出几道题', '课程文档', '代码实操', '练习', '讲义', '资料',
                 '选择题', '填空题', '判断题', '简答题', '题目', '拓展阅读', '阅读',
                 '编程题', '生成编程', '算法题'],
    'general': [],
}


class MasterAgent:
    def __init__(self):
        self._name = 'MasterAgent'
        self._agents: Dict[str, BaseAgent] = {}
        self._profile_agent = None
        self._register_default_agents()

    def _register_default_agents(self):
        agent_classes = [
            ('ProfileAgent', ProfileAgent),
            ('ResourceAgent', ResourceAgent),
            ('PathPlanningAgent', PathPlanningAgent),
            ('RecommendAgent', RecommendAgent),
            ('HintAgent', HintAgent),
            ('ErrorAnalysisAgent', ErrorAnalysisAgent),
        ]

        for agent_name, agent_cls in agent_classes:
            try:
                if agent_name == 'ProfileAgent':
                    self._profile_agent = agent_cls()
                    self.register(self._profile_agent)
                else:
                    self.register(agent_cls())
                logger.info(f"{agent_name} registered")
            except Exception as e:
                logger.warning(f"Failed to register {agent_name}: {e}")

    @property
    def name(self) -> str:
        return self._name

    def register(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        logger.info(f"Agent '{agent.name}' registered to MasterAgent")

    def unregister(self, agent_name: str):
        if agent_name in self._agents:
            del self._agents[agent_name]

    def classify_intent(self, message: str) -> str:
        if not message:
            return 'general'
        message_lower = message.lower()
        best_intent = 'general'
        best_score = 0
        for intent, keywords in INTENT_KEYWORDS.items():
            if not keywords:
                continue
            score = 0
            for kw in keywords:
                if kw.lower() in message_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent

    def load_user_profile(self, user_id: int) -> Dict[str, Any]:
        if self._profile_agent is None:
            return {}
        try:
            return self._profile_agent._load_profile_from_neo4j(user_id)
        except Exception as e:
            logger.warning(f"Failed to load user profile for {user_id}: {e}")
            return {}

    def get_clean_user_profile(self, user_id: int) -> Dict[str, Any]:
        if self._profile_agent is None:
            return {}
        try:
            return self._profile_agent.get_clean_profile(user_id)
        except Exception as e:
            logger.warning(f"Failed to get clean profile for {user_id}: {e}")
            return {}

    def handle_message(self, user_id: int, message: str,
                        extra_context: Dict[str, Any] = None) -> Dict[str, Any]:
        user_profile = self.load_user_profile(user_id)

        # 只有通用对话才检查 onboarding，显式请求的 agent 功能不拦截
        forced_agent = (extra_context or {}).get('agent_type', '')

        if not forced_agent and user_profile and not user_profile.get('_onboarding_complete', True):
            agent = self._agents.get('ProfileAgent')
            if agent:
                result = agent.process_onboarding_answer(user_id, message)
                return {
                    'agent': 'ProfileAgent',
                    'intent': 'profile_onboarding',
                    'onboarding_complete': result.get('onboarding_complete', False),
                    'question': result.get('question'),
                    'message': result.get('message'),
                    'step': result.get('step'),
                    'total_steps': result.get('total_steps'),
                }

        # 显式指定 agent 时直接路由
        if forced_agent:
            agent = self._agents.get(forced_agent)
            if agent:
                context = {
                    'user_id': user_id,
                    'message': message,
                    'intent': forced_agent.lower(),
                    'existing_profile': user_profile,
                }
                if extra_context:
                    context.update(extra_context)
                try:
                    result = agent.run(context)
                    result['agent'] = forced_agent
                    result['intent'] = forced_agent.lower()
                    return result
                except Exception:
                    logger.exception(f"{forced_agent} failed, falling back to intent classification")

        intent = self.classify_intent(message)
        logger.info(f"MasterAgent: user_id={user_id}, intent={intent}, message='{message[:80]}...'")

        context = {
            'user_id': user_id,
            'message': message,
            'intent': intent,
            'existing_profile': user_profile,
        }
        if extra_context:
            context.update(extra_context)

        if intent == 'profile':
            agent = self._agents.get('ProfileAgent')
            if agent:
                try:
                    result = agent.run(context)
                    result['agent'] = 'ProfileAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', ['正在调取学习画像...', '正在分析做题数据...'])
                    return result
                except Exception:
                    logger.exception("ProfileAgent failed, falling back to LLM")

        if intent == 'resource':
            agent = self._agents.get('ResourceAgent')
            if agent:
                try:
                    resource_type = self._classify_resource_type(message)
                    context['resource_type'] = resource_type
                    result = agent.run(context)
                    result['agent'] = 'ResourceAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', self._get_resource_steps(resource_type, result.get('topic', '')))
                    return result
                except Exception:
                    logger.exception("ResourceAgent failed, falling back to LLM")

        if intent == 'recommend':
            agent = self._agents.get('RecommendAgent')
            if agent:
                try:
                    result = agent.run(context)
                    result['agent'] = 'RecommendAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', ['正在调取学习画像...', '正在知识图谱召回...', '正在协同过滤召回...', '正在GNN神经网络召回...', '正在DeepFM精排...'])
                    return result
                except Exception:
                    logger.exception("RecommendAgent failed, falling back to LLM")

        if intent == 'hint':
            agent = self._agents.get('HintAgent')
            if agent:
                try:
                    result = agent.run(context)
                    result['agent'] = 'HintAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', ['正在分析题目...', '正在检索解题资料...', '正在生成渐进式提示...'])
                    return result
                except Exception:
                    logger.exception("HintAgent failed, falling back to LLM")

        if intent == 'analyze_error':
            agent = self._agents.get('ErrorAnalysisAgent')
            if agent:
                try:
                    result = agent.run(context)
                    result['agent'] = 'ErrorAnalysisAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', ['正在获取提交历史...', '正在分析错误类型...', '正在诊断薄弱知识点...'])
                    return result
                except Exception:
                    logger.exception("ErrorAnalysisAgent failed, falling back to LLM")

        if intent == 'learning_path':
            agent = self._agents.get('PathPlanningAgent')
            if agent:
                try:
                    result = agent.run(context)
                    result['agent'] = 'PathPlanningAgent'
                    result['intent'] = intent
                    result.setdefault('thinking_steps', ['正在识别薄弱知识点...', '正在分析知识图谱依赖关系...', '正在计算最优学习路径...'])
                    if result.get('success') and result.get('path_plan'):
                        result['display_type'] = 'path_plan'
                    return result
                except Exception:
                    logger.exception("PathPlanningAgent failed, falling back to LLM")

        fallback = self._fallback_llm(user_id, message, intent)
        if fallback:
            return fallback

        return {
            'agent': 'MasterAgent',
            'intent': intent,
            'message': '抱歉，暂时无法处理你的请求，请稍后再试。',
        }

    def _fallback_llm(self, user_id: int, message: str, intent: str) -> Dict[str, Any]:
        try:
            from aiChat.utils import ask_ai

            prompt = self._build_fallback_prompt(user_id, message)
            answer = ask_ai(prompt)
            return {
                'agent': 'LLM',
                'intent': intent,
                'message': answer,
                'thinking_steps': ['正在调用大模型处理...'],
            }
        except Exception:
            logger.exception("LLM fallback failed")
            return None

    def _build_fallback_prompt(self, user_id: int, message: str) -> str:
        """构建通用对话 prompt（含用户画像），供同步/流式 fallback 复用"""
        profile_summary = ''
        if user_id > 0:
            profile = self.get_clean_user_profile(user_id) or {}
            strengths = profile.get('strength_topics', [])
            weaks = profile.get('weak_topics', [])
            goals = profile.get('learning_goals', '')
            parts = []
            if strengths:
                parts.append(f"强项: {', '.join(strengths)}")
            if weaks:
                parts.append(f"弱项: {', '.join(weaks)}")
            if goals:
                parts.append(f"学习目标: {goals}")
            if parts:
                profile_summary = '\n'.join(parts)

        prompt_parts = ['你是一个专业的编程竞赛辅导助手，服务于一个在线判题系统（OJ）的学生用户。']
        if profile_summary:
            prompt_parts.append(f'\n学生画像:\n{profile_summary}')
        prompt_parts.append(f'\n学生的问题：{message}')
        prompt_parts.append('\n请给出有帮助的、结构清晰的回答。如果问题与编程无关，可以自由回答。')
        return '\n'.join(prompt_parts)

    def stream_fallback_llm(self, user_id: int, message: str):
        """流式版本的通用对话 fallback，yield 文本 chunk。"""
        from aiChat.utils import ask_deepseek_stream
        prompt = self._build_fallback_prompt(user_id, message)
        for chunk in ask_deepseek_stream(prompt):
            yield chunk

    def handle_submission_event(self, user_id: int,
                                  event: Dict[str, Any]) -> Dict[str, Any]:
        if self._profile_agent is None:
            return {'success': False, 'error': 'ProfileAgent not available'}

        profile = self.load_user_profile(user_id)
        context = {
            'user_id': user_id,
            'existing_profile': profile,
            'submission_event': event,
            'intent': 'profile',
            'message': '',
        }
        return self._profile_agent.run(context)

    def _classify_resource_type(self, message: str) -> str:
        m = message.lower()
        if any(kw in m for kw in ['思维导图', '导图', '画图', '脑图', 'mindmap', 'mermaid']):
            return 'mindmap'
        if any(kw in m for kw in ['编程题', '生成编程题', 'OJ题目', '算法题']):
            return 'coding_problem'
        if any(kw in m for kw in ['出题', '题目', '生成题目', '出几道题', '练习', '选择题', '填空题', '判断题']):
            return 'exercise'
        if any(kw in m for kw in ['阅读', '材料', '资料', '推荐', '书单', '清单']):
            return 'reading'
        if any(kw in m for kw in ['代码', '案例', '实操', '示例', 'code', '编程', '实现']):
            return 'code_example'
        return 'lecture'

    def _get_resource_steps(self, resource_type: str, topic: str) -> list:
        steps = ['正在检索相关资料...', '正在分析知识图谱结构...']
        if resource_type == 'lecture':
            steps.append('正在生成课程讲解文档...')
        elif resource_type == 'mindmap':
            steps.append('正在生成思维导图...')
        elif resource_type == 'exercise':
            steps.append(f'正在生成练习题...')
        elif resource_type == 'reading':
            steps.append('正在整理阅读清单...')
        elif resource_type == 'code_example':
            steps.append('正在生成代码案例...')
        elif resource_type == 'coding_problem':
            steps.append('正在生成编程题目...')
        else:
            steps.append('正在生成资源...')
        return steps

    @property
    def agents(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)

    @property
    def profile_agent(self):
        return self._profile_agent


master_agent = MasterAgent()
