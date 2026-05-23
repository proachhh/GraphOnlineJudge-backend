import json
import re
import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from utils.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)

RESOURCE_PROMPTS = {
    'lecture': """# 角色
你是一个资深编程教师。请根据提供的参考资料，生成一份结构严谨、深入浅出的课程讲解文档。

# 参考资料 (RAG检索结果)
{rag_context}

# 知识图谱上下文
{kg_context}

# 格式要求
请严格按照以下 Markdown 格式输出课程讲解文档：

## 1. 概念引入
- 从实际场景出发，解释这个知识点是什么
- 为什么要学这个

## 2. 核心原理
- 深入讲解核心思想
- 配合图示说明（用文字描述）

## 3. 关键步骤 / 算法流程
- 分步骤详细说明
- 每个步骤给出解释

## 4. 复杂度分析
- 时间复杂度
- 空间复杂度

## 5. 注意事项与常见误区
- 3-5 个要点

## 6. 小结
- 一句话总结核心思想
- 后续学习建议

请直接输出 Markdown 文档，不要输出其他解释。""",

    'mindmap': """# 角色
你是一个知识图谱可视化专家。

# 知识图谱数据
{kg_context}

# 关联知识点 (从向量检索中获取)
{rag_context}

# 任务
根据上面的知识点数据，生成一个完整的 Mermaid mindmap 代码。
必须包含：
- 中心节点：用户查询的目标知识点
- 一级分支：前置知识、核心内容、后继知识、相关概念、典型应用
- 二级节点：每个分支下的具体知识点或概念
- 如果有题目数据，加入"相关题目"分支

# 输出格式
只输出 Mermaid 代码，放在 ```mermaid 代码块中。
格式示例:
```mermaid
mindmap
  root((目标知识点))
    前置知识
      知识A
      知识B
    核心内容
      概念1
      概念2
    后继知识
      知识C
    相关概念
      知识D
```
""",

    'exercise': """# 角色
你是一个专业的编程竞赛出题官。

# 参考资料 (RAG检索结果)
{rag_context}

# 知识点信息
目标知识点: {topic}
难度: {difficulty}

# 任务
请基于以上资料，生成以下 4 种题型的练习题各 1 道：

1. **选择题**: 4 个选项，注明正确答案
2. **填空题**: 留下 2-3 个空让填写
3. **判断题**: 给出陈述，判断对错，注明正确答案
4. **简答题**: 需要简要文字回答

每道题都需要包含：
- 题目描述
- 正确答案
- 解析（简要说明为什么）

# 输出格式
严格按照以下 JSON 格式输出，不要输出其他文字：
```json
{{
  "choice": {{
    "stem": "题目描述",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A",
    "explanation": "解析"
  }},
  "fill_blank": {{
    "stem": "题目描述，用 (1) (2) 表示填空",
    "answers": ["答案1", "答案2"],
    "explanation": "解析"
  }},
  "true_false": {{
    "stem": "陈述内容",
    "answer": true,
    "explanation": "解析"
  }},
  "short_answer": {{
    "stem": "问题描述",
    "reference_answer": "参考答案要点",
    "explanation": "解析"
  }}
}}
```
""",

    'reading': """# 角色
你是一个编程学习资源推荐官。

# 参考资料 (RAG检索结果)
{rag_context}

# 知识图谱上下文
{kg_context}

# 任务
基于参考资料，整理一份推荐阅读清单。每篇推荐包含：
- 标题
- 推荐理由（1-2句）
- 内容摘要（2-3句）
- 适合人群

# 输出格式
严格按照以下 JSON 格式输出，不要输出其他文字：
```json
{{
  "readings": [
    {{
      "title": "阅读材料标题",
      "reason": "推荐理由",
      "summary": "内容摘要",
      "audience": "适合人群"
    }}
  ],
  "reading_order": "建议阅读顺序说明"
}}
```

至少推荐 3 篇，最多 5 篇。""",

    'code_example': """# 角色
你是一个资深程序员兼技术导师。

# 参考资料 (RAG检索结果)
{rag_context}

# 知识点信息
目标知识点: {topic}

# 任务
生成一份完整的代码实操案例，包含：
1. 问题场景描述
2. 完整的可运行代码（Python）
3. 逐步讲解（每段代码后附带解释）
4. 运行示例（输入/输出）
5. 变体练习建议

# 格式要求
请用 Markdown 格式输出，代码放在 ```python 代码块中。

结构如下：
## 场景描述
## 完整代码
## 逐步讲解
## 运行示例
## 变体练习
"""
}


class ResourceAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='ResourceAgent')

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_message = context.get('message', '')
        resource_type = context.get('resource_type', 'lecture')
        topic = context.get('topic', '')
        difficulty = context.get('difficulty', 'Mid')

        if not topic:
            topic = self._extract_topic_from_message(user_message)

        rag_docs = self._rag_search(topic, user_message)
        rag_context = self._format_rag_context(rag_docs)

        kg_context = self._fetch_kg_context(topic)

        try:
            if resource_type == 'lecture':
                return self._generate_lecture(topic, rag_context, kg_context)
            elif resource_type == 'mindmap':
                return self._generate_mindmap(topic, rag_context, kg_context)
            elif resource_type == 'exercise':
                return self._generate_exercises(topic, difficulty, rag_context)
            elif resource_type == 'reading':
                return self._generate_reading(topic, rag_context, kg_context)
            elif resource_type == 'code_example':
                return self._generate_code_example(topic, rag_context)
            else:
                return {
                    'success': False,
                    'error': f'未知的资源类型: {resource_type}',
                    'resource_type': resource_type,
                }
        except Exception as e:
            logger.exception(f"ResourceAgent generation failed for {resource_type}")
            return {
                'success': False,
                'error': f'生成失败: {str(e)}',
                'resource_type': resource_type,
            }

    def _extract_topic_from_message(self, message: str) -> str:
        STOP_WORDS = {
            '课程讲解', '生成', '文档', '给我', '一个', '一些', '一份',
            '帮我', '请', '的', '了', '是', '在', '和', '与', '或',
            '讲解', '资料', '题目', '出题', '出几道', '生成资料', '生成题目',
            '思维导图', '导图', '阅读材料', '代码案例', '代码实操', '实操',
            '阅读', '拓展', '整理', '笔记', '讲稿', '课程', '讲义', '练习',
            '选择题', '填空题', '判断题', '简答题',
        }

        patterns = [
            r'关于\s*[「「](.+?)[」」]',
            r'知识点\s*[:：]\s*(.+?)(?:[，,。]|$)',
            r'讲解\s*(.+?)(?:[，,。]|$)',
            r'(.+?)\s*的\s*(?:讲解|资料|题目|思维导图|代码)',
        ]
        for p in patterns:
            m = re.search(p, message)
            if m:
                topic = m.group(1).strip()
                if topic not in STOP_WORDS and len(topic) >= 2:
                    return topic

        cleaned = message
        for sw in sorted(STOP_WORDS, key=len, reverse=True):
            cleaned = cleaned.replace(sw, ' ')

        parts = [w.strip() for w in re.split(r'\s+', cleaned) if len(w.strip()) > 1]
        for part in parts[:5]:
            clean = re.sub(r'[^\u4e00-\u9fff\w]', '', part)
            if len(clean) >= 2:
                return clean

        return message[:20]

    def _rag_search(self, topic: str, query: str) -> List[Dict]:
        try:
            from utils.vector_store import get_vector_store
            store = get_vector_store('oj_documents')
            if not store.is_ready:
                logger.warning("VectorStore not ready, skipping RAG")
                return []
            search_query = f"{topic} {query}"[:200]
            results = store.search(search_query, top_k=5)
            logger.info(f"RAG search for '{topic}': found {len(results)} documents")
            return results
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return []

    def _format_rag_context(self, docs: List[Dict]) -> str:
        if not docs:
            return '（暂无相关参考资料）'
        parts = []
        for i, doc in enumerate(docs):
            content = doc.get('content', '')[:1500]
            meta = doc.get('metadata', {})
            source = meta.get('source', '')
            tags = meta.get('tags', '')
            score = doc.get('score', 0)
            header = f"【参考资料 {i+1}】"
            if source:
                header += f" 来源: {source}"
            if tags:
                header += f" 标签: {tags}"
            header += f" 相关度: {score}"
            parts.append(f"{header}\n{content}\n")
        return '\n---\n'.join(parts)

    def _fetch_kg_context(self, topic: str) -> str:
        try:
            client = neo4j_client

            query = """
            MATCH (t:Topic {name: $topic})
            OPTIONAL MATCH (t)-[:PREREQUISITE_OF]->(next:Topic)
            OPTIONAL MATCH (prev:Topic)-[:PREREQUISITE_OF]->(t)
            OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Topic)
            RETURN t.name AS topic,
                   collect(DISTINCT prev.name) AS prerequisites,
                   collect(DISTINCT next.name) AS successors,
                   collect(DISTINCT related.name) AS related_topics
            """
            result = client.run_query(query, {'topic': topic})

            if not result:
                return f'目标知识点: {topic}\n（知识图谱中暂无此知识点数据）'

            r = result[0]
            lines = [f"目标知识点: {r.get('topic', topic)}"]

            prereqs = [x for x in (r.get('prerequisites') or []) if x]
            if prereqs:
                lines.append(f"前置知识点: {', '.join(prereqs)}")

            successors = [x for x in (r.get('successors') or []) if x]
            if successors:
                lines.append(f"后继知识点: {', '.join(successors)}")

            related = [x for x in (r.get('related_topics') or []) if x]
            if related:
                lines.append(f"相关知识点: {', '.join(related)}")

            problem_query = """
            MATCH (t:Topic {name: $topic})<-[:BELONGS_TO]-(p:Problem)
            RETURN p.title AS title, p.difficulty AS difficulty
            ORDER BY p.accepted_number DESC LIMIT 5
            """
            problems = client.run_query(problem_query, {'topic': topic})
            if problems:
                prob_lines = [f"  - {p['title']} (难度: {p.get('difficulty', 'N/A')})" for p in problems]
                lines.append(f"相关题目:\n" + '\n'.join(prob_lines))

            return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"KG query failed for '{topic}': {e}")
            return f'目标知识点: {topic}\n（知识图谱查询失败）'

    def _call_ai(self, prompt: str) -> str:
        try:
            from aiChat.utils import ask_ai
            return ask_ai(prompt)
        except Exception as e:
            logger.error(f"AI call failed: {e}")
            return f"【AI 服务调用失败: {str(e)}】"

    def _generate_lecture(self, topic: str, rag_context: str, kg_context: str) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPTS['lecture'].format(
            rag_context=rag_context,
            kg_context=kg_context,
        )
        response = self._call_ai(prompt)
        return {
            'success': True,
            'agent': 'ResourceAgent',
            'resource_type': 'lecture',
            'intent': 'resource',
            'topic': topic,
            'content': response,
            'display_type': 'markdown',
        }

    def _generate_mindmap(self, topic: str, rag_context: str, kg_context: str) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPTS['mindmap'].format(
            rag_context=rag_context,
            kg_context=kg_context,
        )
        response = self._call_ai(prompt)

        mermaid_code = response
        m = re.search(r'```mermaid\s*(.*?)\s*```', response, re.DOTALL)
        if m:
            mermaid_code = m.group(1).strip()
        else:
            m = re.search(r'```\s*(mindmap.*?)```', response, re.DOTALL)
            if m:
                mermaid_code = m.group(1).strip()

        return {
            'success': True,
            'agent': 'ResourceAgent',
            'resource_type': 'mindmap',
            'intent': 'resource',
            'topic': topic,
            'content': mermaid_code,
            'display_type': 'mermaid',
        }

    def _generate_exercises(self, topic: str, difficulty: str, rag_context: str) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPTS['exercise'].format(
            rag_context=rag_context,
            topic=topic,
            difficulty=difficulty,
        )
        response = self._call_ai(prompt)

        exercises = self._parse_json_response(response)
        if not exercises:
            exercises = {
                'choice': {'stem': '解析失败', 'options': [], 'answer': '', 'explanation': ''},
                'fill_blank': {'stem': '解析失败', 'answers': [], 'explanation': ''},
                'true_false': {'stem': '解析失败', 'answer': False, 'explanation': ''},
                'short_answer': {'stem': '解析失败', 'reference_answer': '', 'explanation': ''},
            }

        return {
            'success': True,
            'agent': 'ResourceAgent',
            'resource_type': 'exercise',
            'intent': 'resource',
            'topic': topic,
            'difficulty': difficulty,
            'content': exercises,
            'display_type': 'exercises',
        }

    def _generate_reading(self, topic: str, rag_context: str, kg_context: str) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPTS['reading'].format(
            rag_context=rag_context,
            kg_context=kg_context,
        )
        response = self._call_ai(prompt)
        data = self._parse_json_response(response)
        return {
            'success': True,
            'agent': 'ResourceAgent',
            'resource_type': 'reading',
            'intent': 'resource',
            'topic': topic,
            'content': data if data else {'readings': [], 'reading_order': ''},
            'display_type': 'reading_list',
        }

    def _generate_code_example(self, topic: str, rag_context: str) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPTS['code_example'].format(
            rag_context=rag_context,
            topic=topic,
        )
        response = self._call_ai(prompt)

        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)

        return {
            'success': True,
            'agent': 'ResourceAgent',
            'resource_type': 'code_example',
            'intent': 'resource',
            'topic': topic,
            'content': response,
            'code_blocks': code_blocks,
            'display_type': 'code_example',
        }

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        try:
            if '```json' in response:
                start = response.index('```json') + 7
                end = response.index('```', start)
                response = response[start:end]
            elif '```' in response:
                start = response.index('```') + 3
                end = response.index('```', start)
                response = response[start:end]
            response = response.strip()
            if response.startswith('{'):
                return json.loads(response)
            else:
                return json.loads('{' + response.split('{', 1)[1].rsplit('}', 1)[0] + '}')
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"JSON parse failed: {e}")
            return None
