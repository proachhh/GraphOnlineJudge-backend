from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q, FloatField, F
from django.db.models.functions import Cast
from problem.models import ProblemTag
from submission.models import Submission, JudgeStatus
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import TruncDate
from utils.neo4j_client import neo4j_client
import logging

logger = logging.getLogger(__name__)


@login_required
def learning_stats(request):
    user = request.user
    total_submissions = Submission.objects.filter(user_id=user.id).count()
    total_ac = Submission.objects.filter(user_id=user.id, result=JudgeStatus.ACCEPTED).count()
    accuracy = round(total_ac / total_submissions * 100, 1) if total_submissions else 0

    tag_stats = []
    user_problem_ids = Submission.objects.filter(
        user_id=user.id
    ).values_list('problem_id', flat=True).distinct()

    tags_with_data = ProblemTag.objects.filter(problem__id__in=user_problem_ids).annotate(
        total=Count('problem__submission', filter=Q(problem__submission__user_id=user.id)),
        ac=Count(
            'problem__submission',
            filter=Q(problem__submission__user_id=user.id) &
            Q(problem__submission__result=JudgeStatus.ACCEPTED)
        )
    ).filter(total__gt=0).order_by('-total')[:8]

    for tag in tags_with_data:
        acc_rate = round(tag.ac / tag.total * 100, 1) if tag.total else 0
        tag_stats.append({
            'tag_name': tag.name,
            'total': tag.total,
            'ac': tag.ac,
            'accuracy': acc_rate,
        })

    tag_stats.sort(key=lambda x: x['accuracy'])

    beat_percent = get_beat_percent(user)

    from django.db.models.functions import Lower
    lang_stats = []
    raw_lang_groups = Submission.objects.filter(user_id=user.id).values('language').annotate(
        total=Count('id'),
        ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
    )

    merged_langs = {}
    for group in raw_lang_groups:
        raw_lang = group['language']
        import re
        normalized = re.sub(r'[^a-zA-Z]', '', raw_lang).lower()
        alias_map = {
            'cpp': 'c++',
            'cplusplus': 'c++',
            'py': 'python',
            'python3': 'python',
            'js': 'javascript',
            'node': 'javascript',
            'golang': 'go',
        }
        display_lang = alias_map.get(normalized, raw_lang)
        key = display_lang.lower()
        if key not in merged_langs:
            merged_langs[key] = {
                'lang_name': display_lang,
                'total': 0,
                'ac': 0,
            }
        merged_langs[key]['total'] += group['total']
        merged_langs[key]['ac'] += group['ac']

    for key, data in merged_langs.items():
        total = data['total']
        ac = data['ac']
        acc_rate = round(ac / total * 100, 1) if total else 0
        lang_stats.append({
            'lang_name': data['lang_name'],
            'total': total,
            'ac': ac,
            'accuracy': acc_rate,
        })
    lang_stats.sort(key=lambda x: x['accuracy'])

    data = {
        'total_submissions': total_submissions,
        'total_ac': total_ac,
        'accuracy': accuracy,
        'beat_percent': beat_percent,
        'tags': tag_stats,
        'lang_stats': lang_stats,
    }
    return JsonResponse(data)


def get_beat_percent(user):
    user_stats = Submission.objects.filter(user_id=user.id).aggregate(
        total_sub=Count('id'),
        total_ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
    )
    if user_stats['total_sub'] == 0:
        return 0.0
    user_accuracy = user_stats['total_ac'] / user_stats['total_sub'] * 100
    all_users_stats = Submission.objects.values('user_id').annotate(
        total_sub=Count('id'),
        total_ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
    ).filter(total_sub__gt=0).annotate(
        accuracy=Cast(F('total_ac'), FloatField()) / Cast(F('total_sub'), FloatField()) * 100
    ).order_by('-accuracy')
    higher_count = all_users_stats.filter(accuracy__gt=user_accuracy).count()
    total_users = all_users_stats.count()
    if total_users == 0:
        return 0.0
    beat = (total_users - higher_count) / total_users * 100
    return round(beat, 1)


@login_required
def learning_trend(request):
    days = int(request.GET.get('days', 7))
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=days - 1)
    submissions = Submission.objects.filter(
        user_id=request.user.id,
        create_time__date__gte=start_date,
        create_time__date__lte=end_date
    ).annotate(date=TruncDate('create_time')).values('date').annotate(
        total=Count('id'),
        ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
    ).order_by('date')
    date_range = [start_date + timedelta(days=i) for i in range(days)]
    result = []
    for d in date_range:
        item = next((s for s in submissions if s['date'] == d), None)
        if item and item['total'] > 0:
            rate = round(item['ac'] / item['total'] * 100, 1)
        else:
            rate = 0
        result.append({
            'date': d.strftime('%m/%d'),
            'accuracy': rate
        })
    return JsonResponse({'trend': result})


def knowledge_graph_overview(request):
    nodes_query = """
    MATCH (t:Topic)
    RETURN DISTINCT t.name AS name
    LIMIT 200
    """
    nodes_result = neo4j_client.run_query(nodes_query)
    nodes = [{'name': record['name']} for record in nodes_result]

    edges_query = """
    MATCH (t1:Topic)-[:PREREQUISITE_OF]->(t2:Topic)
    RETURN DISTINCT t1.name AS source, t2.name AS target
    LIMIT 500
    """
    edges_result = neo4j_client.run_query(edges_query)
    edges = [{'source': record['source'], 'target': record['target']} for record in edges_result]

    return JsonResponse({'nodes': nodes, 'edges': edges})
