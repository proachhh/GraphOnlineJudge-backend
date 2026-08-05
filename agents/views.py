import json
import logging
import time

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from agents.master_agent import master_agent
from agents.profile_agent import ONBOARDING_DIMENSION_ORDER

logger = logging.getLogger(__name__)


def _get_safe_encoder():
    from django.core.serializers.json import DjangoJSONEncoder
    class SafeEncoder(DjangoJSONEncoder):
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            return super().default(obj)
    return SafeEncoder

def _check_login(request):
    if not request.user.is_authenticated:
        return None
    return request.user.id


@csrf_exempt
def agent_chat(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_message = data.get('message', '').strip()
    if not user_message:
        return JsonResponse({'error': 'message is required'}, status=400)

    resource_type = data.get('resource_type', '')
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'Mid')
    submission_id = data.get('submission_id', '')

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    try:
        extra_context = {}
        agent_type = data.get('agent_type', '')
        if agent_type:
            extra_context['agent_type'] = agent_type
        if resource_type:
            extra_context['resource_type'] = resource_type
        if topic:
            extra_context['topic'] = topic
        if difficulty:
            extra_context['difficulty'] = difficulty
        if submission_id:
            extra_context['submission_id'] = submission_id

        result = master_agent.handle_message(user_id or 0, user_message, extra_context)
        return JsonResponse({
            'success': True,
            'data': result,
        }, encoder=_get_safe_encoder())
    except Exception as e:
        logger.exception(f"MasterAgent handle_message failed")
        return JsonResponse({
            'success': False,
            'error': str(e),
        }, status=500)


@csrf_exempt
def profile_init(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    if user_id is None:
        return JsonResponse({'error': '请先登录'}, status=401)

    answer = data.get('answer', '').strip()
    action = data.get('action', 'start')

    profile_agent = master_agent.profile_agent
    if profile_agent is None:
        return JsonResponse({'error': 'ProfileAgent 未就绪'}, status=500)

    try:
        if action == 'start':
            result = profile_agent.start_onboarding(user_id)
        elif action == 'answer':
            if not answer:
                return JsonResponse({'error': 'answer is required for action=answer'}, status=400)
            result = profile_agent.process_onboarding_answer(user_id, answer)
        elif action == 'skip':
            result = profile_agent.process_onboarding_answer(
                user_id, '（用户选择跳过）'
            )
        elif action == 'status':
            profile = profile_agent._load_profile_from_neo4j(user_id)
            answered = profile.get('_onboarding_answers', {})

            has_profile_data = bool(
                profile.get('strength_topics') or
                profile.get('weak_topics') or
                profile.get('recommended_focus')
            )
            is_complete = profile.get('_onboarding_complete', False) or has_profile_data

            result = {
                'onboarding_complete': is_complete,
                'answered_count': len(answered),
                'total': len(ONBOARDING_DIMENSION_ORDER),
                'profile': profile_agent.get_clean_profile(user_id) if is_complete else {},
                'message': '画像已存在，无需重新引导' if is_complete and not profile.get('_onboarding_complete') else None,
            }
        elif action == 'reset':
            from utils.neo4j_client import neo4j_client
            neo4j_client.run_query("""
                MATCH (u:User {user_id: $user_id})
                REMOVE u.profile_onboarding_answers,
                       u.profile_onboarding_current_dim,
                       u.profile_onboarding_complete,
                       u.profile_knowledge_mastery,
                       u.profile_strength_topics,
                       u.profile_weak_topics,
                       u.profile_coding_style,
                       u.profile_learning_pace,
                       u.profile_recommended_focus
            """, {'user_id': user_id})
            result = {'onboarding_complete': False, 'message': '画像已重置'}
        else:
            return JsonResponse({'error': f'未知的 action: {action}'}, status=400)

        return JsonResponse({
            'success': True,
            'data': result,
        }, encoder=_get_safe_encoder())
    except Exception as e:
        logger.exception(f"Profile init failed")
        return JsonResponse({
            'success': False,
            'error': str(e),
        }, status=500)


@csrf_exempt
def agent_recommend(request):
    user_id = _check_login(request)
    if user_id is None:
        return JsonResponse({'error': '请先登录'}, status=401)
    limit = int(request.GET.get('limit', 5))
    offset = int(request.GET.get('offset', 0))

    agent = master_agent.agents.get('RecommendAgent')
    if agent is None:
        return JsonResponse({'error': 'RecommendAgent 未就绪'}, status=500)

    try:
        context = {
            'user_id': user_id,
            'message': '推荐题目',
            'limit': limit,
            'offset': offset,
            'existing_profile': master_agent.load_user_profile(user_id),
        }
        result = agent.run(context)
        if result.get('success'):
            return JsonResponse({
                'recommendations': result.get('recommendations', []),
                'total': result.get('total', 0),
            }, encoder=_get_safe_encoder())
        else:
            return JsonResponse({'error': result.get('error', '推荐失败')}, status=500)
    except Exception as e:
        logger.exception(f"RecommendAgent failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def agent_chat_stream(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_message = data.get('message', '').strip()
    if not user_message:
        return JsonResponse({'error': 'message is required'}, status=400)

    submission_id = data.get('submission_id', '')
    resource_type = data.get('resource_type', '')
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'Mid')
    agent_type = data.get('agent_type', '')

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    extra_context = {}
    if agent_type:
        extra_context['agent_type'] = agent_type
    if resource_type:
        extra_context['resource_type'] = resource_type
    if topic:
        extra_context['topic'] = topic
    if difficulty:
        extra_context['difficulty'] = difficulty
    if submission_id:
        extra_context['submission_id'] = submission_id

    def sse_event(event_type, data):
        return f"data: {json.dumps({'event': event_type, **data}, ensure_ascii=False, cls=_get_safe_encoder())}\n\n"

    def event_stream():
        # 1. 先做意图分类（不调 LLM），发送思考步骤
        intent = master_agent.classify_intent(user_message)
        thinking_steps = _get_thinking_steps(intent)
        for step in thinking_steps:
            yield sse_event('step', {'text': step})
            time.sleep(0.3)

        forced_agent = extra_context.get('agent_type', '')

        # 2a. 通用对话（无指定 agent 且 intent=general）：真流式调用 LLM
        if not forced_agent and intent == 'general':
            yield sse_event('step', {'text': '正在生成回答...'})
            full_text = ''
            try:
                for chunk in master_agent.stream_fallback_llm(user_id or 0, user_message):
                    full_text += chunk
                    yield sse_event('chunk', {'text': chunk})
            except Exception as e:
                logger.exception("stream_fallback_llm failed")
                if not full_text:
                    full_text = f'抱歉，生成回答时出错：{e}'
                    yield sse_event('chunk', {'text': full_text})
            yield sse_event('done', {})
            yield sse_event('result', {'data': {
                'agent': 'LLM',
                'intent': 'general',
                'message': full_text,
            }})
            return

        # 2b. 特定 agent：调 handle_message 获取完整结果（内部调 ask_ai 非流式）
        result = master_agent.handle_message(user_id or 0, user_message, extra_context)

        # 如果 result 里也有 thinking_steps 但没有预设的，补上
        if not result.get('thinking_steps'):
            result['thinking_steps'] = thinking_steps

        yield sse_event('done', {})
        time.sleep(0.1)

        # 3. 提取文本内容，逐块发送（agent 结果含结构化数据，前端会用 result 事件替换为卡片）
        text = _get_result_text(result)
        if text:
            import re
            chunks = _split_for_stream(text)
            for chunk in chunks:
                yield sse_event('chunk', {'text': chunk})
                time.sleep(0.02)

        # 4. 发送最终结构化结果（前端会替换为格式化卡片）
        yield sse_event('result', {'data': result})

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


def _get_thinking_steps(intent):
    steps_map = {
        'profile': ['正在调取学习画像...', '正在分析做题数据...', '正在生成能力评估...'],
        'recommend': ['正在调取学习画像...', '正在知识图谱召回...', '正在协同过滤召回...', '正在 GNN 神经网络召回...', '正在 DeepFM 精排...'],
        'hint': ['正在分析题目...', '正在检索解题资料...', '正在生成渐进式提示...'],
        'analyze_error': ['正在获取提交历史...', '正在分析错误类型...', '正在诊断薄弱知识点...'],
        'learning_path': ['正在识别薄弱知识点...', '正在分析知识图谱依赖关系...', '正在计算最优学习路径...'],
        'resource': ['正在检索教学资料...', '正在生成内容...'],
    }
    return steps_map.get(intent, ['正在思考中...'])


def _get_result_text(result):
    """从 agent 结果中提取用于流式显示的文本"""
    # 优先取直接文本字段
    for key in ('message', 'analysis', 'hint', 'content'):
        text = result.get(key)
        if text and isinstance(text, str) and len(text) > 20:
            return text
    return ''


def _split_for_stream(text):
    """将文本拆分成适合流式显示的块"""
    import re
    chunks = []
    # 先按段落分
    parts = re.split(r'(\n\n|\n)', text)
    for part in parts:
        if not part.strip():
            chunks.append(part)
            continue
        # 段落太长则按句子分
        if len(part) > 80:
            sentences = re.split(r'([。！？!?\n])', part)
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                if sentence.strip():
                    # 句子太长再按长度切
                    if len(sentence) > 30:
                        for j in range(0, len(sentence), 20):
                            chunks.append(sentence[j:j + 20])
                    else:
                        chunks.append(sentence)
        else:
            chunks.append(part)
    return chunks


@csrf_exempt
def agent_immersion(request):
    user_id = _check_login(request)
    if user_id is None:
        return JsonResponse({'error': '请先登录'}, status=401)
    limit = int(request.GET.get('limit', 10))
    offset = int(request.GET.get('offset', 0))

    agent = master_agent.agents.get('RecommendAgent')
    if agent is None:
        return JsonResponse({'error': 'RecommendAgent 未就绪'}, status=500)

    try:
        context = {
            'user_id': user_id,
            'message': '沉浸式刷题',
            'limit': limit,
            'offset': offset,
            'existing_profile': master_agent.load_user_profile(user_id),
        }
        result = agent.run(context)
        if result.get('success'):
            return JsonResponse({
                'problems': result.get('recommendations', []),
                'total': len(result.get('recommendations', [])),
                'current_index': offset,
            }, encoder=_get_safe_encoder())
        else:
            return JsonResponse({'error': result.get('error', '推荐失败')}, status=500)
    except Exception as e:
        logger.exception(f"Immersion recommend failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def agent_learning_path(request):
    user_id = _check_login(request)
    if user_id is None:
        return JsonResponse({'error': '请先登录'}, status=401)
    start_topic = request.GET.get('start_topic', '')
    target_topic = request.GET.get('target_topic', '')

    agent = master_agent.agents.get('PathPlanningAgent')
    if agent is None:
        return JsonResponse({'error': 'PathPlanningAgent 未就绪'}, status=500)

    try:
        context = {
            'user_id': user_id,
            'message': '规划学习路径',
            'existing_profile': master_agent.load_user_profile(user_id),
        }
        if start_topic:
            context['start_topic'] = start_topic
        if target_topic:
            context['target_topic'] = target_topic
        result = agent.run(context)
        if result.get('success'):
            return JsonResponse(result, encoder=_get_safe_encoder())
        else:
            return JsonResponse({'error': result.get('error', '路径规划失败')}, status=500)
    except Exception as e:
        logger.exception(f"PathPlanningAgent failed")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def agent_profile(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    user_id = _check_login(request)
    if user_id is None:
        return JsonResponse({'error': '请先登录'}, status=401)

    profile_agent = master_agent.profile_agent
    if profile_agent is None:
        return JsonResponse({'error': 'ProfileAgent 未就绪'}, status=500)

    try:
        raw = profile_agent.get_clean_profile(user_id)

        DEFAULT_STRINGS = {
            'knowledge_mastery': '提交更多题目后，系统将自动为你生成知识点掌握度分析',
            'coding_style': '提交更多代码后，系统将为你分析编码风格',
            'learning_pace': '提交更多题目后，系统将为你分析学习节奏',
            'recommended_focus': '提交更多题目后，系统将为你推荐重点提升方向',
        }

        profile = {}
        profile['knowledge_mastery'] = raw.get('knowledge_mastery') or DEFAULT_STRINGS['knowledge_mastery']
        profile['strength_topics'] = raw.get('strength_topics') if raw.get('strength_topics') else []
        profile['weak_topics'] = raw.get('weak_topics') if raw.get('weak_topics') else []
        profile['coding_style'] = raw.get('coding_style') or DEFAULT_STRINGS['coding_style']
        profile['learning_pace'] = raw.get('learning_pace') or DEFAULT_STRINGS['learning_pace']
        profile['recommended_focus'] = raw.get('recommended_focus') or DEFAULT_STRINGS['recommended_focus']
        profile['background'] = raw.get('background', '')
        profile['current_courses'] = raw.get('current_courses', '')
        profile['weak_areas'] = raw.get('weak_areas', '')
        profile['learning_goals'] = raw.get('learning_goals', '')
        profile['learning_style'] = raw.get('learning_style', '')
        profile['weekly_hours'] = raw.get('weekly_hours', '')

        return JsonResponse({
            'success': True,
            'profile': profile,
        })
    except Exception as e:
        logger.exception(f"ProfileAgent get_profile failed")
        return JsonResponse({
            'success': False,
            'error': str(e),
        }, status=500)
