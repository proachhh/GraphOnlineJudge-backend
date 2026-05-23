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
