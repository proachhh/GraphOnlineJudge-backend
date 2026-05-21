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

    user_id = None
    if request.user.is_authenticated:
        user_id = request.user.id

    try:
        result = master_agent.handle_message(user_id or 0, user_message)
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
