import json
import logging
import time

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from agents.master_agent import master_agent
from agents.profile_agent import ONBOARDING_DIMENSION_ORDER

logger = logging.getLogger(__name__)


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

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    try:
        extra_context = {}
        if resource_type:
            extra_context['resource_type'] = resource_type
        if topic:
            extra_context['topic'] = topic
        if difficulty:
            extra_context['difficulty'] = difficulty

        result = master_agent.handle_message(user_id or 0, user_message, extra_context)
        return JsonResponse({
            'success': True,
            'data': result,
        })
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
        })
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
            })
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

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    def sse_event(event_type, data):
        return f"data: {json.dumps({'event': event_type, **data}, ensure_ascii=False)}\n\n"

    def event_stream():
        result = master_agent.handle_message(user_id or 0, user_message)
        thinking_steps = result.get('thinking_steps', [])

        for step in thinking_steps:
            yield sse_event('step', {'text': step})
            time.sleep(0.3)

        yield sse_event('done', {})
        time.sleep(0.15)

        yield sse_event('result', {'data': result})

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


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
            })
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
            return JsonResponse(result)
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
