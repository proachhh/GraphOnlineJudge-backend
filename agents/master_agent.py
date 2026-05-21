import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from agents.profile_agent import ProfileAgent

logger = logging.getLogger(__name__)

INTENT_KEYWORDS = {
    'profile': ['画像', '分析我的学习', '我的水平', '我学得怎么样', '学习报告', '我的能力',
                '我的强项', '我的弱项', '知识掌握', '学习评估', '当前能力', '我的学习情况'],
    'recommend': ['推荐', '下一题', '做题建议', '刷题', '学什么', '推荐题目',
                  '有什么题目', '什么题适合我'],
    'hint': ['提示', '怎么做', '思路', '解题方向', '帮我理解', '这道题', '参考答案',
             '解答', '不会做', '教教我'],
    'analyze_error': ['为什么错了', '哪里错了', '错误分析', '帮我看看', '提交失败',
                      '运行错误', '编译错误', '超时', '为什么没过', '错在哪'],
    'learning_path': ['学习路径', '学习路线', '怎么学', '学习计划', '进阶路线',
                      '先学什么', '规划', '路径', '路线图'],
    'general': [],
}


class MasterAgent:
    def __init__(self):
        self._name = 'MasterAgent'
        self._agents: Dict[str, BaseAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        try:
            self.register(ProfileAgent())
            logger.info("ProfileAgent registered")
        except Exception as e:
            logger.warning(f"Failed to register ProfileAgent: {e}")

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

    def handle_message(self, user_id: int, message: str) -> Dict[str, Any]:
        intent = self.classify_intent(message)
        logger.info(f"MasterAgent: user_id={user_id}, intent={intent}, message='{message[:80]}...'")

        context = {
            'user_id': user_id,
            'message': message,
            'intent': intent,
        }

        if intent == 'profile':
            agent = self._agents.get('ProfileAgent')
            if agent:
                result = agent.run(context)
                result['agent'] = 'ProfileAgent'
                result['intent'] = intent
                return result

        return {
            'agent': 'MasterAgent',
            'intent': intent,
            'message': f'意图识别: {intent}。该功能即将上线。',
        }

    @property
    def agents(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)


master_agent = MasterAgent()
