import functools
import json
import logging
import re
import difflib

from django.db import transaction
from django.db.models import Avg, Count, Q
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from aiChat.utils import ask_ai_stream
from account.models import User
from utils.neo4j_client import neo4j_client

from .models import (BossAnswer, BossExam, BossQuestion, BossSubmission,
                     ExerciseAnswer, ExerciseQuestion, ExerciseSet, ExerciseSubmission)

logger = logging.getLogger(__name__)

SUBJECTIVE_TYPES = ['code', 'short_answer']


# ---------------------------------------------------------------------------
# Decorators (function-based, return JSON)
# ---------------------------------------------------------------------------
def login_required(func):
    """Require an authenticated user. Returns JSON 401 when not logged in."""

    @functools.wraps(func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': '请先登录'}, status=401)
        if request.user.is_disabled:
            return JsonResponse({'error': '账号已被禁用'}, status=403)
        return func(request, *args, **kwargs)

    return wrapper


def teacher_required(func):
    """Require a Teacher or Admin role. Returns JSON 401/403 otherwise."""

    @functools.wraps(func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': '请先登录'}, status=401)
        if request.user.is_disabled:
            return JsonResponse({'error': '账号已被禁用'}, status=403)
        if not request.user.is_teacher_or_admin():
            return JsonResponse({'error': '需要教师或管理员权限'}, status=403)
        return func(request, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_body(request):
    try:
        return json.loads(request.body or '{}'), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JsonResponse({'error': 'Invalid JSON'}, status=400)


def _grade_choice(question, answer):
    """Compare answer with correct_answer. Returns (is_correct, score)."""
    correct = (question.correct_answer or '').strip()
    given = (answer or '').strip()
    if correct and given:
        is_correct = given.upper() == correct.upper()
    else:
        is_correct = False
    return is_correct, (question.score if is_correct else 0)


def _sse(event_type, data):
    return f"data: {json.dumps({'event': event_type, **data}, ensure_ascii=False)}\n\n"


def _serialize_question(q, include_answer=False):
    """Serialize a question for API response.

    When include_answer is False the correct_answer / explanation fields are
    excluded so students cannot peek at the answers.
    """
    data = {
        'id': q.id,
        'order': q.order,
        'question_type': q.question_type,
        'content': q.content,
        'choices': q.choices or [],
        'score': q.score,
        'problem': None,
    }
    if q.problem_id:
        problem = q.problem
        data['problem'] = {
            'id': q.problem_id,
            '_id': getattr(problem, '_id', None),
            'title': getattr(problem, 'title', None),
        }
    if hasattr(q, 'topic'):
        data['topic'] = q.topic
    if include_answer:
        data['correct_answer'] = q.correct_answer
        data['explanation'] = q.explanation
    return data


def _build_grade_prompt(question, answer_text, topic):
    parts = [
        "你是编程教学助教，请批改学生的作答。",
        "",
        f"【知识点】{topic or '未指定'}",
        f"【题目】{question.content}",
    ]
    if question.choices:
        opts = '\n'.join(f"{c.get('key', '')}. {c.get('text', '')}" for c in question.choices)
        parts.append(f"【选项】\n{opts}")
    parts.append(f"【学生答案】{answer_text or '（未作答）'}")
    parts.append("")
    parts.append("请严格按以下 JSON 格式返回（不要输出其他内容）：")
    parts.append(
        '{"is_correct": true或false, "score": 0到%d的整数, "feedback": "批改反馈说明"}' % question.score
    )
    return '\n'.join(parts)


def _parse_grade_result(text, max_score):
    """Extract (is_correct, score, feedback) from the AI response text."""
    match = re.search(r'\{[\s\S]*\}', text or '')
    if match:
        try:
            data = json.loads(match.group(0))
            is_correct = data.get('is_correct')
            if isinstance(is_correct, str):
                is_correct = is_correct.strip().lower() in ('true', '1', 'correct', '正确')
            elif isinstance(is_correct, (int, float)):
                is_correct = bool(is_correct)
            try:
                score = int(data.get('score', 0))
            except (TypeError, ValueError):
                score = 0
            score = max(0, min(score, max_score or 0))
            feedback = data.get('feedback') or data.get('explanation') or text
            return is_correct, score, feedback
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None, 0, text


def _ai_grade_stream(question, answer, topic):
    """Generator yielding SSE event strings for AI grading of a subjective
    question. `answer` is the student's answer record (ExerciseAnswer /
    BossAnswer). The graded result is persisted into `answer`."""
    answer_text = answer.answer or ''
    prompt = _build_grade_prompt(question, answer_text, topic)
    full_text = ''
    try:
        for chunk in ask_ai_stream(prompt):
            full_text += chunk
            yield _sse('chunk', {'question_id': question.id, 'text': chunk})
    except Exception as e:
        logger.exception("ask_ai_stream failed during grading")
        full_text = f"批改过程中出错：{e}"
        yield _sse('error', {'question_id': question.id, 'error': str(e)})

    is_correct, score, feedback = _parse_grade_result(full_text, question.score)

    answer.ai_feedback = feedback
    answer.is_correct = is_correct
    answer.score = score
    if hasattr(answer, 'graded_at'):
        answer.graded_at = timezone.now()
    answer.save()

    yield _sse('graded', {
        'question_id': question.id,
        'is_correct': is_correct,
        'score': score,
        'feedback': feedback,
    })


def _stream_response(generator):
    response = StreamingHttpResponse(generator, content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


def _get_owned_set(user, set_id, model=ExerciseSet):
    """Return an ExerciseSet/BossExam owned by `user` (admins own everything)."""
    qs = model.objects.all()
    if not user.is_admin_role():
        qs = qs.filter(created_by=user)
    return qs.filter(id=set_id).first()


# ===========================================================================
# Student - ExerciseSet
# ===========================================================================
@csrf_exempt
@login_required
def exercise_set_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    qs = ExerciseSet.objects.filter(is_published=True)
    topic = request.GET.get('topic')
    if topic:
        qs = qs.filter(topic=topic)
    difficulty = request.GET.get('difficulty')
    if difficulty:
        qs = qs.filter(difficulty=difficulty)

    qs = qs.annotate(
        question_count=Count('questions', distinct=True),
        submission_count=Count('submissions', distinct=True),
    ).order_by('-create_time')

    submitted_ids = set(
        ExerciseSubmission.objects.filter(user=request.user).values_list('exercise_set_id', flat=True)
    )

    data = []
    for s in qs:
        data.append({
            'id': s.id,
            'title': s.title,
            'description': s.description,
            'topic': s.topic,
            'difficulty': s.difficulty,
            'question_count': s.question_count,
            'submission_count': s.submission_count,
            'time_limit': s.time_limit,
            'max_attempts': s.max_attempts,
            'show_score': s.show_score,
            'show_ranking': s.show_ranking,
            'has_submitted': s.id in submitted_ids,
        })
    return JsonResponse({'data': data})


@csrf_exempt
@login_required
def exercise_set_detail(request, set_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        s = ExerciseSet.objects.get(id=set_id, is_published=True)
    except ExerciseSet.DoesNotExist:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    questions = s.questions.select_related('problem').order_by('order', 'id')
    return JsonResponse({
        'id': s.id,
        'title': s.title,
        'description': s.description,
        'topic': s.topic,
        'difficulty': s.difficulty,
        'time_limit': s.time_limit,
        'max_attempts': s.max_attempts,
        'show_score': s.show_score,
        'show_ranking': s.show_ranking,
        'create_time': s.create_time,
        'questions': [_serialize_question(q, include_answer=False) for q in questions],
    })


@csrf_exempt
@login_required
def exercise_start(request, set_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        s = ExerciseSet.objects.get(id=set_id, is_published=True)
    except ExerciseSet.DoesNotExist:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    # Count completed attempts (submitted/graded, not in_progress)
    completed_count = ExerciseSubmission.objects.filter(
        exercise_set=s, user=request.user
    ).exclude(status='in_progress').count()
    if s.max_attempts and completed_count >= s.max_attempts:
        return JsonResponse({'error': f'已达到最大答题次数({s.max_attempts}次)'}, status=403)
    # Reuse an existing in_progress submission or create a new one
    sub = ExerciseSubmission.objects.filter(
        exercise_set=s, user=request.user, status='in_progress'
    ).first()
    if sub is None:
        sub = ExerciseSubmission.objects.create(
            exercise_set=s, user=request.user, status='in_progress'
        )
    questions = s.questions.select_related('problem').order_by('order', 'id')
    return JsonResponse({
        'submission_id': sub.id,
        'status': sub.status,
        'create_time': sub.create_time,
        'time_limit': s.time_limit,
        'max_attempts': s.max_attempts,
        'completed_count': completed_count,
        'questions': [_serialize_question(q, include_answer=False) for q in questions],
    })


@csrf_exempt
@login_required
def exercise_submit(request, set_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        s = ExerciseSet.objects.get(id=set_id, is_published=True)
    except ExerciseSet.DoesNotExist:
        return JsonResponse({'error': '试题集不存在'}, status=404)

    data, err = _parse_json_body(request)
    if err:
        return err
    answers = data.get('answers') or []

    # Prefer an in_progress submission; otherwise create a new one
    sub = ExerciseSubmission.objects.filter(
        exercise_set=s, user=request.user, status='in_progress'
    ).first()
    if sub is None:
        # Safety check: enforce max_attempts when creating a new submission
        completed_count = ExerciseSubmission.objects.filter(
            exercise_set=s, user=request.user
        ).exclude(status='in_progress').count()
        if s.max_attempts and completed_count >= s.max_attempts:
            return JsonResponse({'error': f'已达到最大答题次数({s.max_attempts}次)'}, status=403)
        sub = ExerciseSubmission.objects.create(exercise_set=s, user=request.user, status='submitted')

    total = 0
    pending_ai = 0
    with transaction.atomic():
        for item in answers:
            qid = item.get('question_id')
            ans = item.get('answer')
            try:
                q = ExerciseQuestion.objects.get(id=qid, exercise_set=s)
            except ExerciseQuestion.DoesNotExist:
                continue
            obj, _ = ExerciseAnswer.objects.update_or_create(
                submission=sub, question=q, defaults={'answer': ans}
            )
            if q.question_type == 'choice':
                is_correct, score = _grade_choice(q, ans)
                obj.is_correct = is_correct
                obj.score = score
                obj.ai_feedback = None
                obj.save()
                total += score
            else:
                obj.is_correct = None
                obj.score = 0
                obj.ai_feedback = None
                obj.save()
                pending_ai += 1

        sub.total_score = total
        sub.status = 'graded' if pending_ai == 0 else 'submitted'
        sub.save()

    return JsonResponse({
        'submission_id': sub.id,
        'total_score': sub.total_score,
        'status': sub.status,
        'pending_ai_grading': pending_ai,
    })


@csrf_exempt
@login_required
def exercise_report(request, sub_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        sub = ExerciseSubmission.objects.select_related('exercise_set').get(
            id=sub_id, user=request.user
        )
    except ExerciseSubmission.DoesNotExist:
        return JsonResponse({'error': '作答记录不存在'}, status=404)

    answers = {a.question_id: a for a in sub.answers.select_related('question', 'question__problem')}
    questions = sub.exercise_set.questions.select_related('problem').order_by('order', 'id')
    show_score = sub.exercise_set.show_score
    q_list = []
    for q in questions:
        a = answers.get(q.id)
        item = _serialize_question(q, include_answer=True)
        item['answer'] = a.answer if a else None
        item['is_correct'] = a.is_correct if a else None
        item['answer_score'] = a.score if a else 0
        item['ai_feedback'] = a.ai_feedback if a else None
        q_list.append(item)

    return JsonResponse({
        'submission_id': sub.id,
        'exercise_set': {
            'id': sub.exercise_set.id,
            'title': sub.exercise_set.title,
            'topic': sub.exercise_set.topic,
            'difficulty': sub.exercise_set.difficulty,
            'show_ranking': sub.exercise_set.show_ranking,
        },
        'total_score': sub.total_score if show_score else None,
        'show_score': show_score,
        'status': sub.status,
        'create_time': sub.create_time,
        'questions': q_list,
    })


@csrf_exempt
@login_required
def exercise_ranking(request, set_id):
    """Get ranking for an exercise set (respects show_ranking flag)."""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        s = ExerciseSet.objects.get(id=set_id, is_published=True)
    except ExerciseSet.DoesNotExist:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    if not s.show_ranking:
        return JsonResponse({'error': '排名未开放'}, status=403)

    subs = ExerciseSubmission.objects.filter(
        exercise_set=s, status='graded'
    ).select_related('user').order_by('-total_score', 'create_time')

    ranking = []
    my_rank = None
    for idx, sub in enumerate(subs, 1):
        is_me = sub.user_id == request.user.id
        entry = {
            'rank': idx,
            'username': sub.user.username,
            'score': sub.total_score if s.show_score else None,
            'create_time': sub.create_time,
            'is_me': is_me,
        }
        if is_me:
            my_rank = idx
        ranking.append(entry)

    return JsonResponse({
        'ranking': ranking[:50],
        'my_rank': my_rank,
        'total_participants': subs.count(),
    })


@csrf_exempt
@login_required
def exercise_grade_stream(request, sub_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        sub = ExerciseSubmission.objects.select_related('exercise_set').get(
            id=sub_id, user=request.user
        )
    except ExerciseSubmission.DoesNotExist:
        return JsonResponse({'error': '作答记录不存在'}, status=404)

    topic = sub.exercise_set.topic
    pending = list(
        sub.answers.select_related('question', 'question__problem').filter(
            question__question_type__in=SUBJECTIVE_TYPES, ai_feedback__isnull=True
        )
    )

    def event_stream():
        yield _sse('start', {'submission_id': sub.id, 'count': len(pending)})
        total = sub.total_score
        for ans in pending:
            q = ans.question
            for sse in _ai_grade_stream(q, ans, topic):
                yield sse
            total += ans.score
            yield _sse('score_update', {'question_id': q.id, 'total_score': total})

        sub.total_score = total
        remaining = sub.answers.filter(
            question__question_type__in=SUBJECTIVE_TYPES, ai_feedback__isnull=True
        ).count()
        if remaining == 0:
            sub.status = 'graded'
        sub.save()
        yield _sse('done', {
            'submission_id': sub.id,
            'total_score': sub.total_score,
            'status': sub.status,
        })

    return _stream_response(event_stream())


# ===========================================================================
# Student - BossExam
# ===========================================================================
@csrf_exempt
@login_required
def boss_exam_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    qs = BossExam.objects.filter(is_published=True).annotate(
        question_count=Count('questions', distinct=True),
        submission_count=Count('submissions', distinct=True),
    ).order_by('-create_time')

    submitted_ids = set(
        BossSubmission.objects.filter(user=request.user).values_list('exam_id', flat=True)
    )

    data = []
    for e in qs:
        data.append({
            'id': e.id,
            'title': e.title,
            'description': e.description,
            'topic_area': e.topic_area or [],
            'boss_topic': e.boss_topic,
            'difficulty': e.difficulty,
            'question_count': e.question_count,
            'submission_count': e.submission_count,
            'time_limit': e.time_limit,
            'passing_score': e.passing_score,
            'has_submitted': e.id in submitted_ids,
        })
    return JsonResponse({'data': data})


@csrf_exempt
@login_required
def boss_exam_detail(request, exam_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        e = BossExam.objects.get(id=exam_id, is_published=True)
    except BossExam.DoesNotExist:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)
    questions = e.questions.select_related('problem').order_by('order', 'id')
    return JsonResponse({
        'id': e.id,
        'title': e.title,
        'description': e.description,
        'topic_area': e.topic_area or [],
        'boss_topic': e.boss_topic,
        'difficulty': e.difficulty,
        'time_limit': e.time_limit,
        'passing_score': e.passing_score,
        'create_time': e.create_time,
        'questions': [_serialize_question(q, include_answer=False) for q in questions],
    })


@csrf_exempt
@login_required
def boss_exam_start(request, exam_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        e = BossExam.objects.get(id=exam_id, is_published=True)
    except BossExam.DoesNotExist:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)
    sub, created = BossSubmission.objects.get_or_create(
        exam=e, user=request.user, status='in_progress'
    )
    questions = e.questions.select_related('problem').order_by('order', 'id')
    return JsonResponse({
        'submission_id': sub.id,
        'status': sub.status,
        'created': created,
        'questions': [_serialize_question(q, include_answer=False) for q in questions],
    })


@csrf_exempt
@login_required
def boss_exam_submit(request, exam_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        e = BossExam.objects.get(id=exam_id, is_published=True)
    except BossExam.DoesNotExist:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)

    data, err = _parse_json_body(request)
    if err:
        return err
    answers = data.get('answers') or []

    sub = BossSubmission.objects.filter(
        exam=e, user=request.user
    ).order_by('-create_time').first()
    if sub is None:
        sub = BossSubmission.objects.create(exam=e, user=request.user, status='submitted')

    total = 0
    pending_ai = 0
    with transaction.atomic():
        for item in answers:
            qid = item.get('question_id')
            ans = item.get('answer')
            try:
                q = BossQuestion.objects.get(id=qid, exam=e)
            except BossQuestion.DoesNotExist:
                continue
            obj, _ = BossAnswer.objects.update_or_create(
                submission=sub, question=q, defaults={'answer': ans}
            )
            if q.question_type == 'choice':
                is_correct, score = _grade_choice(q, ans)
                obj.is_correct = is_correct
                obj.score = score
                obj.ai_feedback = None
                obj.save()
                total += score
            else:
                obj.is_correct = None
                obj.score = 0
                obj.ai_feedback = None
                obj.save()
                pending_ai += 1

        sub.total_score = total
        sub.status = 'graded' if pending_ai == 0 else 'submitted'
        sub.completed_at = timezone.now()
        sub.passed = total >= (e.passing_score or 0)
        sub.save()

    return JsonResponse({
        'submission_id': sub.id,
        'total_score': sub.total_score,
        'status': sub.status,
        'passed': sub.passed,
        'pending_ai_grading': pending_ai,
    })


@csrf_exempt
@login_required
def boss_exam_report(request, sub_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        sub = BossSubmission.objects.select_related('exam').get(id=sub_id, user=request.user)
    except BossSubmission.DoesNotExist:
        return JsonResponse({'error': '作答记录不存在'}, status=404)

    answers = {a.question_id: a for a in sub.answers.select_related('question', 'question__problem')}
    questions = sub.exam.questions.select_related('problem').order_by('order', 'id')
    q_list = []
    for q in questions:
        a = answers.get(q.id)
        item = _serialize_question(q, include_answer=True)
        item['answer'] = a.answer if a else None
        item['is_correct'] = a.is_correct if a else None
        item['answer_score'] = a.score if a else 0
        item['ai_feedback'] = a.ai_feedback if a else None
        q_list.append(item)

    return JsonResponse({
        'submission_id': sub.id,
        'exam': {
            'id': sub.exam.id,
            'title': sub.exam.title,
            'boss_topic': sub.exam.boss_topic,
            'topic_area': sub.exam.topic_area or [],
            'difficulty': sub.exam.difficulty,
            'passing_score': sub.exam.passing_score,
        },
        'total_score': sub.total_score,
        'status': sub.status,
        'passed': sub.passed,
        'ai_evaluation': sub.ai_evaluation,
        'create_time': sub.create_time,
        'completed_at': sub.completed_at,
        'questions': q_list,
    })


@csrf_exempt
@login_required
def boss_grade_stream(request, sub_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        sub = BossSubmission.objects.select_related('exam').get(id=sub_id, user=request.user)
    except BossSubmission.DoesNotExist:
        return JsonResponse({'error': '作答记录不存在'}, status=404)

    exam = sub.exam
    pending = list(
        sub.answers.select_related('question', 'question__problem').filter(
            question__question_type__in=SUBJECTIVE_TYPES, ai_feedback__isnull=True
        )
    )

    def event_stream():
        yield _sse('start', {'submission_id': sub.id, 'count': len(pending)})
        total = sub.total_score
        graded_summary = []
        for ans in pending:
            q = ans.question
            topic = q.topic or exam.boss_topic
            for sse in _ai_grade_stream(q, ans, topic):
                yield sse
            total += ans.score
            graded_summary.append({
                'topic': q.topic,
                'score': ans.score,
                'max_score': q.score,
            })
            yield _sse('score_update', {'question_id': q.id, 'total_score': total})

        sub.total_score = total
        sub.passed = total >= (exam.passing_score or 0)
        remaining = sub.answers.filter(
            question__question_type__in=SUBJECTIVE_TYPES, ai_feedback__isnull=True
        ).count()
        if remaining == 0:
            sub.status = 'graded'
        sub.save()

        # Overall AI evaluation of the boss exam performance
        yield _sse('evaluation_start', {})
        eval_prompt = _build_boss_evaluation_prompt(exam, sub, graded_summary)
        eval_text = ''
        try:
            for chunk in ask_ai_stream(eval_prompt):
                eval_text += chunk
                yield _sse('evaluation_chunk', {'text': chunk})
        except Exception as e:
            logger.exception("ask_ai_stream failed during boss evaluation")
            eval_text = f"生成总体评价时出错：{e}"
            yield _sse('error', {'error': str(e)})
        sub.ai_evaluation = eval_text
        sub.save()
        yield _sse('evaluation_done', {'evaluation': eval_text})

        yield _sse('done', {
            'submission_id': sub.id,
            'total_score': sub.total_score,
            'status': sub.status,
            'passed': sub.passed,
        })

    return _stream_response(event_stream())


def _build_boss_evaluation_prompt(exam, sub, graded_summary):
    summary_text = '\n'.join(
        f"- 知识点 {g.get('topic') or '未指定'}：得分 {g.get('score')}/{g.get('max_score')}"
        for g in graded_summary
    ) or '（无主观题）'
    return (
        "你是编程教学助教，请根据学生在 Boss 试卷上的整体作答情况给出总体评价与提升建议。\n\n"
        f"【试卷标题】{exam.title}\n"
        f"【知识收敛节点（Boss）】{exam.boss_topic}\n"
        f"【知识范围】{', '.join(exam.topic_area or [])}\n"
        f"【总分】{sub.total_score}\n"
        f"【及格线】{exam.passing_score}\n"
        f"【是否通过】{'通过' if sub.passed else '未通过'}\n"
        f"【各主观题得分】\n{summary_text}\n\n"
        "请用中文给出：1. 总体评价 2. 薄弱知识点 3. 后续学习建议。"
    )


# ===========================================================================
# Teacher - ExerciseSet
# ===========================================================================
@csrf_exempt
@teacher_required
def teacher_exercise_set_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    qs = ExerciseSet.objects.all()
    if not request.user.is_admin_role():
        qs = qs.filter(created_by=request.user)
    topic = request.GET.get('topic')
    if topic:
        qs = qs.filter(topic=topic)
    qs = qs.annotate(
        question_count=Count('questions', distinct=True),
        submission_count=Count('submissions', distinct=True),
    ).order_by('-create_time')

    data = []
    for s in qs:
        questions = []
        for q in s.questions.order_by('order', 'id'):
            questions.append({
                'question_type': q.question_type,
                'content': q.content,
                'choices': q.choices or [],
                'correct_answer': q.correct_answer or '',
                'score': q.score,
                'explanation': q.explanation or '',
                'problem_id': q.problem_id,
            })
        data.append({
            'id': s.id,
            'title': s.title,
            'description': s.description,
            'topic': s.topic,
            'difficulty': s.difficulty,
            'is_published': s.is_published,
            'time_limit': s.time_limit,
            'max_attempts': s.max_attempts,
            'show_score': s.show_score,
            'show_ranking': s.show_ranking,
            'question_count': s.question_count,
            'submission_count': s.submission_count,
            'create_time': s.create_time,
            'update_time': s.update_time,
            'questions': questions,
        })
    return JsonResponse({'data': data})


def _build_exercise_questions(set_obj, questions):
    for idx, qd in enumerate(questions or []):
        ExerciseQuestion.objects.create(
            exercise_set=set_obj,
            problem_id=qd.get('problem_id'),
            order=qd.get('order', idx),
            question_type=qd.get('question_type') or 'choice',
            content=qd.get('content') or '',
            choices=qd.get('choices') or [],
            correct_answer=qd.get('correct_answer'),
            score=qd.get('score', 10),
            explanation=qd.get('explanation'),
        )


@csrf_exempt
@teacher_required
def teacher_exercise_set_create(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    data, err = _parse_json_body(request)
    if err:
        return err
    title = (data.get('title') or '').strip()
    if not title:
        return JsonResponse({'error': 'title is required'}, status=400)

    with transaction.atomic():
        s = ExerciseSet.objects.create(
            title=title,
            description=data.get('description'),
            topic=(data.get('topic') or '').strip(),
            difficulty=data.get('difficulty') or 'Mid',
            time_limit=data.get('time_limit'),
            max_attempts=data.get('max_attempts'),
            show_score=data.get('show_score', True),
            show_ranking=data.get('show_ranking', True),
            created_by=request.user,
            meta=data.get('meta') or {},
        )
        _build_exercise_questions(s, data.get('questions'))
    return JsonResponse({'success': True, 'id': s.id})


@csrf_exempt
@teacher_required
def teacher_exercise_set_update(request, set_id):
    if request.method != 'PUT':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    s = _get_owned_set(request.user, set_id, ExerciseSet)
    if s is None:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    data, err = _parse_json_body(request)
    if err:
        return err

    with transaction.atomic():
        if 'title' in data:
            s.title = (data.get('title') or '').strip() or s.title
        if 'description' in data:
            s.description = data.get('description')
        if 'topic' in data:
            s.topic = data.get('topic') or ''
        if 'difficulty' in data:
            s.difficulty = data.get('difficulty') or 'Mid'
        if 'time_limit' in data:
            s.time_limit = data.get('time_limit')
        if 'max_attempts' in data:
            s.max_attempts = data.get('max_attempts')
        if 'show_score' in data:
            s.show_score = bool(data.get('show_score'))
        if 'show_ranking' in data:
            s.show_ranking = bool(data.get('show_ranking'))
        if 'meta' in data:
            s.meta = data.get('meta') or {}
        if 'is_published' in data:
            s.is_published = bool(data.get('is_published'))
        s.save()

        if 'questions' in data:
            s.questions.all().delete()
            _build_exercise_questions(s, data.get('questions'))
    return JsonResponse({'success': True, 'id': s.id})


@csrf_exempt
@teacher_required
def teacher_exercise_set_delete(request, set_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    s = _get_owned_set(request.user, set_id, ExerciseSet)
    if s is None:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    s.delete()
    return JsonResponse({'success': True})


@csrf_exempt
@teacher_required
def teacher_exercise_set_publish(request, set_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    s = _get_owned_set(request.user, set_id, ExerciseSet)
    if s is None:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    s.is_published = not s.is_published
    s.save(update_fields=['is_published', 'update_time'])
    return JsonResponse({'success': True, 'is_published': s.is_published})


@csrf_exempt
@teacher_required
def teacher_ai_generate_questions(request):
    """Generate a set of questions using AI based on topic and difficulty."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    data, err = _parse_json_body(request)
    if err:
        return err
    topic = (data.get('topic') or '').strip()
    difficulty = data.get('difficulty') or 'Mid'
    count = min(int(data.get('count') or 5), 10)
    q_type = data.get('question_type') or 'choice'

    if not topic:
        return JsonResponse({'error': '请指定知识点'}, status=400)

    diff_text = {'Low': '简单', 'Mid': '中等', 'High': '困难'}.get(difficulty, '中等')
    type_text = '选择题' if q_type == 'choice' else '简答题'

    prompt = (
        f"你是编程教学出题专家。请根据知识点【{topic}】，生成{count}道{diff_text}难度的高质量{type_text}。\n\n"
        f"重要：所有{count}道题目必须全部是{type_text}，不得混合其他题型。\n\n"
        f"请严格按以下JSON数组格式返回（不要输出其他内容，不要输出markdown代码块标记）：\n"
    )
    if q_type == 'choice':
        prompt += (
            '每道题必须包含恰好4个选项(A/B/C/D)，correct_answer为单个大写字母。\n'
            '示例格式：\n'
            '[{"content":"第1题题干","choices":[{"key":"A","text":"选项A内容"},{"key":"B","text":"选项B内容"},'
            '{"key":"C","text":"选项C内容"},{"key":"D","text":"选项D内容"}],"correct_answer":"B",'
            '"score":10,"explanation":"解析说明"},'
            '{"content":"第2题题干","choices":[...同上...],"correct_answer":"A","score":10,"explanation":"解析说明"}]'
        )
    else:
        prompt += (
            '每道题不得包含choices字段，correct_answer为文字参考答案。\n'
            '示例格式：\n'
            '[{"content":"第1题题干","correct_answer":"参考答案内容","score":15,"explanation":"解析说明"},'
            '{"content":"第2题题干","correct_answer":"参考答案内容","score":15,"explanation":"解析说明"}]'
        )

    full_text = ''
    try:
        for chunk in ask_ai_stream(prompt):
            full_text += chunk
    except Exception as e:
        logger.exception("AI generate questions failed")
        return JsonResponse({'error': f'AI生成失败：{e}'}, status=500)

    # Strip markdown code block markers if present
    full_text = re.sub(r'```(?:json)?\s*', '', full_text).strip()

    match = re.search(r'\[[\s\S]*\]', full_text)
    if not match:
        return JsonResponse({'error': 'AI返回格式错误', 'raw': full_text[:500]}, status=500)
    try:
        questions = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'AI返回JSON解析失败', 'raw': full_text[:500]}, status=500)

    # Normalize and enforce question type consistency
    valid_keys = {'A', 'B', 'C', 'D'}
    normalized = []
    for idx, q in enumerate(questions):
        if not isinstance(q, dict) or not q.get('content'):
            continue
        q['question_type'] = q_type
        if 'score' not in q or not q['score']:
            q['score'] = 10 if q_type == 'choice' else 15
        if 'explanation' not in q:
            q['explanation'] = ''
        if 'order' not in q:
            q['order'] = idx

        if q_type == 'choice':
            # Ensure choices is a list of 4 items with keys A/B/C/D
            choices = q.get('choices') or []
            choice_map = {}
            for ch in choices:
                k = str(ch.get('key', '')).strip().upper()
                t = ch.get('text', '').strip()
                if k and t:
                    choice_map[k] = t
            q['choices'] = [
                {'key': k, 'text': choice_map.get(k, f'选项{k}')}
                for k in ['A', 'B', 'C', 'D']
            ]
            # Ensure correct_answer is a single uppercase letter
            ans = str(q.get('correct_answer', '')).strip().upper()
            if ans not in valid_keys:
                ans = 'A'
            q['correct_answer'] = ans
        else:
            # short_answer: remove choices, ensure text answer
            q.pop('choices', None)
            if 'correct_answer' not in q or not q['correct_answer']:
                q['correct_answer'] = ''
        normalized.append(q)

    questions = normalized

    return JsonResponse({'questions': questions})


@csrf_exempt
@teacher_required
def teacher_exercise_submissions(request, set_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    s = _get_owned_set(request.user, set_id, ExerciseSet)
    if s is None:
        return JsonResponse({'error': '试题集不存在'}, status=404)
    subs = s.submissions.select_related('user').order_by('-create_time')
    data = []
    for sub in subs:
        data.append({
            'id': sub.id,
            'user_id': sub.user_id,
            'username': sub.user.username,
            'total_score': sub.total_score,
            'status': sub.status,
            'create_time': sub.create_time,
            'update_time': sub.update_time,
        })
    return JsonResponse({'data': data})


# ===========================================================================
# Teacher - BossExam
# ===========================================================================
@csrf_exempt
@teacher_required
def teacher_boss_exam_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    qs = BossExam.objects.all()
    if not request.user.is_admin_role():
        qs = qs.filter(created_by=request.user)
    qs = qs.annotate(
        question_count=Count('questions', distinct=True),
        submission_count=Count('submissions', distinct=True),
    ).order_by('-create_time')

    data = []
    for e in qs:
        data.append({
            'id': e.id,
            'title': e.title,
            'description': e.description,
            'topic_area': e.topic_area or [],
            'boss_topic': e.boss_topic,
            'difficulty': e.difficulty,
            'is_published': e.is_published,
            'time_limit': e.time_limit,
            'passing_score': e.passing_score,
            'question_count': e.question_count,
            'submission_count': e.submission_count,
            'create_time': e.create_time,
        })
    return JsonResponse({'data': data})


def _build_boss_questions(exam_obj, questions):
    for idx, qd in enumerate(questions or []):
        BossQuestion.objects.create(
            exam=exam_obj,
            problem_id=qd.get('problem_id'),
            order=qd.get('order', idx),
            question_type=qd.get('question_type') or 'choice',
            content=qd.get('content') or '',
            choices=qd.get('choices') or [],
            correct_answer=qd.get('correct_answer'),
            score=qd.get('score', 10),
            topic=qd.get('topic'),
        )


@csrf_exempt
@teacher_required
def teacher_boss_exam_create(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    data, err = _parse_json_body(request)
    if err:
        return err
    title = (data.get('title') or '').strip()
    boss_topic = (data.get('boss_topic') or '').strip()
    if not title:
        return JsonResponse({'error': 'title is required'}, status=400)
    if not boss_topic:
        return JsonResponse({'error': 'boss_topic is required'}, status=400)

    with transaction.atomic():
        e = BossExam.objects.create(
            title=title,
            description=data.get('description'),
            topic_area=data.get('topic_area') or [],
            boss_topic=boss_topic,
            difficulty=data.get('difficulty') or 'Mid',
            created_by=request.user,
            time_limit=data.get('time_limit'),
            passing_score=data.get('passing_score', 60),
        )
        _build_boss_questions(e, data.get('questions'))
    return JsonResponse({'success': True, 'id': e.id})


@csrf_exempt
@teacher_required
def teacher_boss_exam_update(request, exam_id):
    if request.method != 'PUT':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    e = _get_owned_set(request.user, exam_id, BossExam)
    if e is None:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)
    data, err = _parse_json_body(request)
    if err:
        return err

    with transaction.atomic():
        if 'title' in data:
            e.title = (data.get('title') or '').strip() or e.title
        if 'description' in data:
            e.description = data.get('description')
        if 'topic_area' in data:
            e.topic_area = data.get('topic_area') or []
        if 'boss_topic' in data:
            e.boss_topic = (data.get('boss_topic') or '').strip() or e.boss_topic
        if 'difficulty' in data:
            e.difficulty = data.get('difficulty') or 'Mid'
        if 'time_limit' in data:
            e.time_limit = data.get('time_limit')
        if 'passing_score' in data:
            e.passing_score = data.get('passing_score', 60)
        if 'is_published' in data:
            e.is_published = bool(data.get('is_published'))
        e.save()

        if 'questions' in data:
            e.questions.all().delete()
            _build_boss_questions(e, data.get('questions'))
    return JsonResponse({'success': True, 'id': e.id})


@csrf_exempt
@teacher_required
def teacher_boss_exam_delete(request, exam_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    e = _get_owned_set(request.user, exam_id, BossExam)
    if e is None:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)
    e.delete()
    return JsonResponse({'success': True})


@csrf_exempt
@teacher_required
def teacher_boss_exam_publish(request, exam_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    e = _get_owned_set(request.user, exam_id, BossExam)
    if e is None:
        return JsonResponse({'error': 'Boss试卷不存在'}, status=404)
    e.is_published = not e.is_published
    e.save(update_fields=['is_published'])
    return JsonResponse({'success': True, 'is_published': e.is_published})


# ===========================================================================
# Teacher - Students & Topics
# ===========================================================================
@csrf_exempt
@teacher_required
def teacher_student_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    from account.models import AdminType
    # 包含所有非超管用户（普通用户+教师），以及所有有提交记录的用户
    students = User.objects.filter(
        is_disabled=False
    ).exclude(
        admin_type=AdminType.SUPER_ADMIN
    ).annotate(
        exercise_count=Count('exercise_submissions', distinct=True),
    ).order_by('username')

    data = []
    for u in students:
        data.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'exercise_count': u.exercise_count,
        })
    return JsonResponse({'data': data})


@csrf_exempt
@teacher_required
def teacher_student_report(request, user_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        student = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({'error': '学生不存在'}, status=404)

    exercise_subs = list(ExerciseSubmission.objects.filter(user=student).select_related(
        'exercise_set'
    ).order_by('-create_time'))

    scores = [s.total_score or 0 for s in exercise_subs if s.status == 'graded']
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    return JsonResponse({
        'student': {
            'id': student.id,
            'username': student.username,
            'email': student.email,
        },
        'exercise_count': len(exercise_subs),
        'avg_score': avg_score,
        'exercise_submissions': [
            {
                'id': s.id,
                'exercise_set_id': s.exercise_set_id,
                'title': s.exercise_set.title,
                'topic': s.exercise_set.topic,
                'score': s.total_score,
                'status': s.status,
                'created': s.create_time,
            }
            for s in exercise_subs
        ],
    })


@csrf_exempt
@teacher_required
def teacher_student_ai_analysis(request, user_id):
    """Generate AI analysis of a student's learning performance."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        student = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({'error': '学生不存在'}, status=404)

    exercise_subs = list(ExerciseSubmission.objects.filter(user=student).select_related(
        'exercise_set'
    ).order_by('-create_time'))
    scores = [s.total_score or 0 for s in exercise_subs if s.status == 'graded']
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    # Build topic performance summary
    topic_stats = {}
    for s in exercise_subs:
        if s.status != 'graded':
            continue
        topic = s.exercise_set.topic or '未分类'
        if topic not in topic_stats:
            topic_stats[topic] = {'count': 0, 'scores': []}
        topic_stats[topic]['count'] += 1
        topic_stats[topic]['scores'].append(s.total_score or 0)

    topic_lines = []
    for topic, info in topic_stats.items():
        avg = round(sum(info['scores']) / len(info['scores']), 1) if info['scores'] else 0
        topic_lines.append(f"  - {topic}: 作答{info['count']}次, 平均{avg}分")

    prompt = (
        f"你是编程教学分析专家。请根据以下学生的学习数据，给出简明（200字以内）的学习分析建议。\n\n"
        f"【学生】{student.username}\n"
        f"【题集提交数】{len(exercise_subs)}\n"
        f"【平均分】{avg_score}\n"
        f"【各知识点表现】\n" + ('\n'.join(topic_lines) if topic_lines else '  暂无数据') + "\n\n"
        f"请从以下方面分析：\n"
        f"1. 整体学习状态评估\n"
        f"2. 薄弱知识点及建议\n"
        f"3. 下一步学习建议\n"
    )

    full_text = ''
    try:
        for chunk in ask_ai_stream(prompt):
            full_text += chunk
    except Exception as e:
        logger.exception("AI student analysis failed")
        return JsonResponse({'error': f'AI分析失败：{e}'}, status=500)

    return JsonResponse({'analysis': full_text, 'avg_score': avg_score, 'topic_stats': topic_stats})


@csrf_exempt
@login_required
def exercise_topics(request):
    """学生端：从已发布试题集获取知识点列表"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    topics = list(
        ExerciseSet.objects.filter(is_published=True)
        .exclude(topic='')
        .values_list('topic', flat=True)
        .distinct()
    )
    return JsonResponse({'topics': topics})


@csrf_exempt
@teacher_required
def teacher_topic_list(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    topics = []
    try:
        records = neo4j_client.run_query("MATCH (t:Topic) RETURN t.name as name")
        for rec in records:
            name = rec.get('name')
            if name:
                topics.append(name)
    except Exception as e:
        logger.exception("neo4j topic query failed")
        return JsonResponse({'error': f'查询知识点失败：{e}', 'topics': []}, status=500)
    return JsonResponse({'topics': topics})


@csrf_exempt
@teacher_required
def teacher_stats(request):
    """教师端学情统计聚合接口：汇总指标 + 分布 + 趋势 + Top 学生"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    from account.models import AdminType
    from datetime import timedelta

    now = timezone.now()
    fourteen_days_ago = now - timedelta(days=13)

    # ---- 基础范围：仅统计本教师可见数据（admin 可见全部） ----
    set_qs = ExerciseSet.objects.all()
    boss_qs = BossExam.objects.all()
    if not request.user.is_admin_role():
        set_qs = set_qs.filter(created_by=request.user)
        boss_qs = boss_qs.filter(created_by=request.user)

    exercise_subs = ExerciseSubmission.objects.filter(exercise_set__in=set_qs)
    boss_subs = BossSubmission.objects.filter(exam__in=boss_qs)
    graded_exercise = exercise_subs.filter(status='graded')
    graded_boss = boss_subs.filter(status='graded')

    # ---- 汇总指标 ----
    total_students = User.objects.filter(is_disabled=False).exclude(
        admin_type=AdminType.SUPER_ADMIN
    ).count()
    total_sets = set_qs.count()
    published_sets = set_qs.filter(is_published=True).count()
    total_boss = boss_qs.count()
    total_submissions = exercise_subs.count() + boss_subs.count()
    graded_submissions = graded_exercise.count() + graded_boss.count()

    all_scores = list(graded_exercise.values_list('total_score', flat=True)) + \
        list(graded_boss.values_list('total_score', flat=True))
    all_scores = [s or 0 for s in all_scores]
    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
    pass_count = sum(1 for s in all_scores if s >= 60)
    pass_rate = round(pass_count / len(all_scores) * 100, 1) if all_scores else 0

    summary = {
        'total_students': total_students,
        'total_sets': total_sets,
        'published_sets': published_sets,
        'total_boss': total_boss,
        'total_submissions': total_submissions,
        'graded_submissions': graded_submissions,
        'avg_score': avg_score,
        'pass_rate': pass_rate,
    }

    # ---- 成绩分布 (0-59 / 60-69 / 70-79 / 80-89 / 90-100) ----
    buckets = {'0-59': 0, '60-69': 0, '70-79': 0, '80-89': 0, '90-100': 0}
    for s in all_scores:
        if s >= 90:
            buckets['90-100'] += 1
        elif s >= 80:
            buckets['80-89'] += 1
        elif s >= 70:
            buckets['70-79'] += 1
        elif s >= 60:
            buckets['60-69'] += 1
        else:
            buckets['0-59'] += 1
    score_distribution = [{'label': k, 'value': v} for k, v in buckets.items()]

    # ---- 知识点分布（题集数 + 平均分） ----
    topic_agg = set_qs.exclude(topic='').values('topic').annotate(
        set_count=Count('id', distinct=True),
        avg_score=Avg('submissions__total_score'),
    ).order_by('-set_count')[:12]
    topic_distribution = []
    for t in topic_agg:
        topic_distribution.append({
            'topic': t['topic'] or '未分类',
            'set_count': t['set_count'],
            'avg_score': round(t['avg_score'], 1) if t['avg_score'] is not None else 0,
        })

    # ---- 难度分布 ----
    diff_agg = set_qs.values('difficulty').annotate(count=Count('id'))
    diff_map = {d['difficulty'] or 'Mid': d['count'] for d in diff_agg}
    difficulty_distribution = [
        {'label': '简单', 'value': diff_map.get('Low', 0), 'key': 'Low'},
        {'label': '中等', 'value': diff_map.get('Mid', 0), 'key': 'Mid'},
        {'label': '困难', 'value': diff_map.get('High', 0), 'key': 'High'},
    ]

    # ---- 提交趋势（近 14 天每日提交数） ----
    trend_map = {}
    for i in range(14):
        d = (fourteen_days_ago + timedelta(days=i)).date()
        trend_map[d.isoformat()] = 0
    for sub in list(exercise_subs.filter(create_time__gte=fourteen_days_ago)) + \
            list(boss_subs.filter(create_time__gte=fourteen_days_ago)):
        key = sub.create_time.date().isoformat()
        if key in trend_map:
            trend_map[key] += 1
    submission_trend = [{'date': k, 'count': v} for k, v in trend_map.items()]

    # ---- Top 学生（按平均分，至少 1 次已评分提交） ----
    top_students_qs = User.objects.filter(
        exercise_submissions__status='graded',
        exercise_submissions__exercise_set__in=set_qs,
        is_disabled=False,
    ).exclude(admin_type=AdminType.SUPER_ADMIN).annotate(
        sub_count=Count('exercise_submissions', distinct=True),
        avg_score=Avg('exercise_submissions__total_score'),
    ).filter(avg_score__isnull=False).order_by('-avg_score')[:5]
    top_students = []
    for u in top_students_qs:
        top_students.append({
            'id': u.id,
            'username': u.username,
            'sub_count': u.sub_count,
            'avg_score': round(u.avg_score, 1) if u.avg_score is not None else 0,
        })

    # ---- 近期提交 ----
    recent = []
    for s in list(exercise_subs.select_related('user', 'exercise_set').order_by('-create_time')[:8]):
        recent.append({
            'id': s.id,
            'username': s.user.username,
            'title': s.exercise_set.title,
            'type': 'exercise',
            'score': s.total_score,
            'status': s.status,
            'created': s.create_time,
        })
    for s in list(boss_subs.select_related('user', 'exam').order_by('-create_time')[:8]):
        recent.append({
            'id': s.id,
            'username': s.user.username,
            'title': s.exam.title,
            'type': 'boss',
            'score': s.total_score,
            'status': s.status,
            'created': s.create_time,
        })
    recent.sort(key=lambda x: x['created'], reverse=True)
    recent = recent[:8]

    return JsonResponse({
        'summary': summary,
        'score_distribution': score_distribution,
        'topic_distribution': topic_distribution,
        'difficulty_distribution': difficulty_distribution,
        'submission_trend': submission_trend,
        'top_students': top_students,
        'recent_submissions': recent,
    })


# ===========================================================================
# Teacher - Code Plagiarism Check
# ===========================================================================

def _normalize_code(code):
    """Normalize code for comparison: strip comments, extra whitespace, blank lines."""
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # Remove string literals (to focus on structure)
    code = re.sub(r'"[^"]*"', '""', code)
    code = re.sub(r"'[^']*'", "''", code)
    # Collapse whitespace
    lines = []
    for line in code.split('\n'):
        line = line.strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


@csrf_exempt
@teacher_required
def teacher_code_check(request):
    """代码查重：对比指定题目所有 AC 提交的代码相似度"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    problem_id = request.GET.get('problem_id')
    if not problem_id:
        return JsonResponse({'error': '请输入题目 ID'}, status=400)

    from submission.models import Submission, JudgeStatus
    from problem.models import Problem

    # 用户输入的是显示 ID（_id），需转换为数据库内部 ID
    # 只查公开题（contest 为空），避免竞赛题重复 _id
    problem = Problem.objects.filter(_id=str(problem_id).strip(), contest__isnull=True).first()
    if not problem:
        return JsonResponse({'error': f'题目 {problem_id} 不存在'}, status=404)

    # 获取该题目所有 AC 的提交
    submissions = list(
        Submission.objects.filter(
            problem_id=problem.id,
            result=JudgeStatus.ACCEPTED
        ).select_related('problem').order_by('user_id', '-create_time')
    )

    if len(submissions) < 2:
        return JsonResponse({'data': [], 'total': 0, 'message': 'AC 提交不足 2 条，无法查重', 'checked': 0})

    # 预处理代码
    normalized = []
    for sub in submissions:
        norm = _normalize_code(sub.code or '')
        normalized.append({
            'submission_id': sub.id,
            'username': sub.username,
            'user_id': sub.user_id,
            'language': sub.language,
            'code': sub.code or '',
            'normalized': norm,
            'create_time': sub.create_time.strftime('%Y-%m-%d %H:%M'),
        })

    # 两两比较（同语言才比较）
    results = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a = normalized[i]
            b = normalized[j]
            # 跳过同一用户
            if a['user_id'] == b['user_id']:
                continue
            # 只比较同语言
            if a['language'] != b['language']:
                continue
            ratio = difflib.SequenceMatcher(None, a['normalized'], b['normalized']).ratio()
            ratio = round(ratio * 100, 1)
            if ratio >= 60:
                results.append({
                    'user_a': a['username'],
                    'user_b': b['username'],
                    'user_a_id': a['user_id'],
                    'user_b_id': b['user_id'],
                    'submission_a': a['submission_id'],
                    'submission_b': b['submission_id'],
                    'language': a['language'],
                    'similarity': ratio,
                    'create_time_a': a['create_time'],
                    'create_time_b': b['create_time'],
                    'code_a': a['code'],
                    'code_b': b['code'],
                })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return JsonResponse({'data': results, 'total': len(results), 'checked': len(submissions)})
