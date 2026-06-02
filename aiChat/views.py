import json
import time
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from account.decorators import login_required
from .utils import ask_ai, ask_ai_stream


@csrf_exempt
def chat(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        model = data.get('model', 'spark')
        if not user_message:
            return JsonResponse({'error': 'message is required'}, status=400)

        answer = ask_ai(user_message, model=model)
        return JsonResponse({'answer': answer, 'model': model})
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def chat_stream(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        model = data.get('model', 'spark')
        if not user_message:
            return JsonResponse({'error': 'message is required'}, status=400)

        def sse_event(event_type, data_dict):
            return f"data: {json.dumps({'event': event_type, **data_dict}, ensure_ascii=False)}\n\n"

        def event_stream():
            for chunk in ask_ai_stream(user_message, model=model):
                yield sse_event('chunk', {'text': chunk})
                time.sleep(0.01)
            yield sse_event('done', {})

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def analyze_error(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        submission_id = data.get('submission_id', '').strip()
        if not submission_id:
            return JsonResponse({'error': 'submission_id is required'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    from submission.models import Submission, JudgeStatus

    try:
        submission = Submission.objects.get(id=submission_id)
    except Submission.DoesNotExist:
        return JsonResponse({'error': '提交记录不存在'}, status=404)

    result_map = {
        JudgeStatus.WRONG_ANSWER: '答案错误',
        JudgeStatus.COMPILE_ERROR: '编译错误',
        JudgeStatus.CPU_TIME_LIMIT_EXCEEDED: '运行超时',
        JudgeStatus.REAL_TIME_LIMIT_EXCEEDED: '运行超时',
        JudgeStatus.MEMORY_LIMIT_EXCEEDED: '内存超限',
        JudgeStatus.RUNTIME_ERROR: '运行时错误',
    }

    status_text = result_map.get(submission.result, f'未知错误(状态码:{submission.result})')

    error_info = submission.statistic_info.get('err_info', '')
    time_cost = submission.statistic_info.get('time_cost', 'N/A')
    memory_cost = submission.statistic_info.get('memory_cost', 'N/A')

    failed_cases = []
    for item in submission.info.get('data', []):
        if item.get('result') != 0:
            failed_cases.append(item)
    failed_cases = failed_cases[:3]

    prompt = f"""你是一个编程竞赛 OJ 系统的 AI 助教。请分析下面这个「{status_text}」的提交，给出问题诊断和改进建议。

【题目信息】
标题：{submission.problem.title}
难度：{submission.problem.difficulty}
时间限制：{submission.problem.time_limit}ms
内存限制：{submission.problem.memory_limit}MB

【提交状态】
状态：{status_text}
语言：{submission.language}
运行时间：{time_cost}
运行内存：{memory_cost}
编译错误信息：{error_info if error_info else '无'}

【失败测试点】
{json.dumps(failed_cases, ensure_ascii=False) if failed_cases else '暂无详细测试点信息'}

【用户代码】
```{submission.language}
{submission.code[:3000]}
```

请用中文回答，格式如下：
1. **问题诊断**：指出代码中可能的错误原因（1-2句话）
2. **具体分析**：分析代码逻辑问题或边界条件遗漏（2-3个要点）
3. **修复建议**：给出具体的修改方向和示例思路"""

    try:
        answer = ask_ai(prompt)
        return JsonResponse({'analysis': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def problem_hint(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        problem_id = data.get('problem_id', '')
        hint_level = int(data.get('hint_level', 1))
        if not problem_id:
            return JsonResponse({'error': 'problem_id is required'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    from problem.models import Problem

    try:
        problem = Problem.objects.get(_id=problem_id)
    except Problem.DoesNotExist:
        return JsonResponse({'error': '题目不存在'}, status=404)

    tags = [tag.name for tag in problem.tags.all()]

    level_guide = {
        1: '只需要给出大致的解题方向和算法思路，不要说具体步骤',
        2: '给出更具体的算法步骤和关键点，但不要给出代码',
        3: '给出详细的解题步骤和伪代码，并提示易错点',
    }

    prompt = f"""你是一个编程竞赛 OJ 系统的 AI 助教。用户正在做一道题，需要获取解题提示。

【题目信息】
标题：{problem.title}
难度：{problem.difficulty}
标签：{', '.join(tags) if tags else '无'}
时间限制：{problem.time_limit}ms
内存限制：{problem.memory_limit}MB

【题目描述】
{problem.description[:2000]}

【输入说明】
{problem.input_description[:500]}

【输出说明】
{problem.output_description[:500]}

【提示级别】
{level_guide.get(hint_level, level_guide[1])}

请用中文给出提示。不要直接输出完整代码。"""

    try:
        answer = ask_ai(prompt)
        return JsonResponse({'hint': answer, 'hint_level': hint_level})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def learning_advice(request):
    try:
        from submission.models import Submission, JudgeStatus
        from utils.neo4j_client import neo4j_client

        if not request.user.is_authenticated:
            return JsonResponse({'error': '请先登录'}, status=401)

        username = request.user.username

        total_sub = Submission.objects.filter(user_id=request.user.id).count()
        total_ac = Submission.objects.filter(user_id=request.user.id, result=JudgeStatus.ACCEPTED).count()
        accuracy = round(total_ac / total_sub * 100, 1) if total_sub else 0

        recent_subs = list(Submission.objects.filter(
            user_id=request.user.id
        ).order_by('-create_time')[:10].values(
            'problem__title', 'problem__difficulty', 'result', 'create_time'
        ))

        recent_list = []
        for s in recent_subs:
            recent_list.append({
                '题目': s.get('problem__title', ''),
                '难度': s.get('problem__difficulty', ''),
                '结果': '通过' if s.get('result') == 0 else '未通过',
                '时间': s['create_time'].strftime('%m/%d') if s.get('create_time') else '未知'
            })

        weak_topics = []
        query = """
        MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)-[:BELONGS_TO]->(t:Topic)
        WITH t, count(s) AS total, sum(CASE WHEN s.result <> '0' THEN 1 ELSE 0 END) AS wrong
        WHERE total >= 2
        RETURN t.name AS topic, wrong * 1.0 / total AS error_rate, total AS total_sub
        ORDER BY error_rate DESC
        LIMIT 5
        """
        try:
            result = neo4j_client.run_query(query, {'username': username})
            for r in result:
                weak_topics.append({
                    'topic': r['topic'],
                    'error_rate': round(r['error_rate'] * 100, 1),
                    'total': r['total_sub']
                })
        except Exception:
            pass

        strong_topics = []
        query = """
        MATCH (u:User {username: $username})-[:SUBMITTED]->(s:Submission)-[:FOR]->(p:Problem)-[:BELONGS_TO]->(t:Topic)
        WHERE s.result = '0'
        WITH t, count(s) AS ac
        WHERE ac >= 2
        RETURN t.name AS topic, ac
        ORDER BY ac DESC
        LIMIT 5
        """
        try:
            result = neo4j_client.run_query(query, {'username': username})
            for r in result:
                strong_topics.append({'topic': r['topic'], 'ac_count': r['ac']})
        except Exception:
            pass

        prompt = f"""你是一个编程竞赛 OJ 系统的 AI 学习顾问。请根据以下用户数据，生成一份个性化的学习建议。

【用户概况】
总提交数：{total_sub}
通过率：{accuracy}%

【薄弱知识点】
{json.dumps(weak_topics, ensure_ascii=False) if weak_topics else '暂无数据'}

【擅长知识点】
{json.dumps(strong_topics, ensure_ascii=False) if strong_topics else '暂无数据'}

【最近提交】
{json.dumps(recent_list, ensure_ascii=False)}

请用中文给出学习建议，格式如下：
1. **整体评价**：一句话总结当前学习状况
2. **薄弱环节**：指出需要重点攻克的知识点（1-2个）
3. **学习计划**：给出接下来 1-2 周的具体学习建议
4. **刷题方向**：推荐练习的题目类型和难度"""

        answer = ask_ai(prompt)
        return JsonResponse({'advice': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def code_review(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        submission_id = data.get('submission_id', '').strip()
        if not submission_id:
            return JsonResponse({'error': 'submission_id is required'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    from submission.models import Submission

    try:
        submission = Submission.objects.get(id=submission_id)
    except Submission.DoesNotExist:
        return JsonResponse({'error': '提交记录不存在'}, status=404)

    time_cost = submission.statistic_info.get('time_cost', 'N/A')
    memory_cost = submission.statistic_info.get('memory_cost', 'N/A')

    prompt = f"""你是一个编程竞赛 OJ 系统的 AI 代码审查员。请审查以下已通过的代码，给出优化建议。

【题目信息】
标题：{submission.problem.title}
难度：{submission.problem.difficulty}
时间限制：{submission.problem.time_limit}ms

【运行结果】
运行时间：{time_cost}
运行内存：{memory_cost}
语言：{submission.language}

【代码】
```{submission.language}
{submission.code[:3000]}
```

请用中文给出代码审查，格式如下：
1. **时间复杂度分析**：分析当前算法的时间复杂度和空间复杂度
2. **代码质量**：评价代码风格、可读性、变量命名等
3. **优化建议**：给出 1-2 个可改进的方向或更优的解法思路
4. **学习建议**：推荐相关的算法知识或数据结构"""

    try:
        answer = ask_ai(prompt)
        return JsonResponse({'review': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def topic_summary(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        topic_name = data.get('topic', '').strip()
        if not topic_name:
            return JsonResponse({'error': 'topic is required'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    from utils.neo4j_client import neo4j_client

    query = """
    MATCH (t:Topic {name: $topic})<-[:BELONGS_TO]-(p:Problem)
    RETURN p.title AS title, p.difficulty AS difficulty, p.accepted_number AS ac
    ORDER BY p.accepted_number DESC
    LIMIT 15
    """
    try:
        problems = neo4j_client.run_query(query, {'topic': topic_name})
    except Exception as e:
        return JsonResponse({'error': f'查询知识点失败: {str(e)}'}, status=500)

    if not problems:
        return JsonResponse({'error': f'未找到知识点「{topic_name}」的相关题目'}, status=404)

    next_topics = []
    try:
        query2 = """
        MATCH (t:Topic {name: $topic})-[:PREREQUISITE_OF]->(next:Topic)
        RETURN next.name AS name
        """
        next_topics = [r['name'] for r in neo4j_client.run_query(query2, {'topic': topic_name})]
    except Exception:
        pass

    problem_list = [{
        'title': p['title'],
        'difficulty': p['difficulty'],
        'ac': p['ac']
    } for p in problems]

    prompt = f"""你是一个编程竞赛 OJ 系统的 AI 知识导师。请根据以下信息，总结知识点「{topic_name}」的学习要点。

【关联题目】
{json.dumps(problem_list, ensure_ascii=False)}

【后续知识点】
{json.dumps(next_topics, ensure_ascii=False) if next_topics else '暂无'}

请用中文给出知识点总结，格式如下：
1. **概念介绍**：用通俗语言解释这个知识点的核心概念
2. **常见题型**：介绍该知识点的 2-3 种典型应用场景
3. **解题模板**：给出通用的解题框架或代码模板（中文伪代码即可）
4. **易错提醒**：指出学习这个知识点时常见的 2-3 个误区
5. **进阶方向**：推荐下一步可以学习的相关知识点"""

    try:
        answer = ask_ai(prompt)
        return JsonResponse({'summary': answer, 'topic': topic_name})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
