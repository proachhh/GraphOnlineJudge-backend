import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from account.decorators import login_required
from agents.master_agent import master_agent

logger = logging.getLogger(__name__)


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
            result = {
                'onboarding_complete': profile.get('_onboarding_complete', False),
                'answered_count': len(answered),
                'total': 6,
                'profile': profile,
            }
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


@login_required
def agent_recommend(request):
    user_id = request.user.id
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


@login_required
def agent_immersion(request):
    user_id = request.user.id
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


@login_required
def agent_learning_path(request):
    user_id = request.user.id
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
