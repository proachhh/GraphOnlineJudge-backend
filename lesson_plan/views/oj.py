from django.db import transaction

from account.decorators import login_required
from utils.api import APIView
from lesson_plan.models import LessonPlan, LessonPlanProblem
from lesson_plan.serializers import LessonPlanSerializer, LessonPlanDetailSerializer
from problem.models import Problem
from submission.models import Submission, JudgeStatus
from aiChat.utils import ask_ai_stream


class LessonPlanAPI(APIView):
    def get(self, request):
        lesson_plan_id = request.GET.get("id")
        if lesson_plan_id:
            try:
                lesson_plan = LessonPlan.objects.select_related("created_by").get(
                    id=lesson_plan_id, visible=True
                )
                return self.success(LessonPlanDetailSerializer(lesson_plan).data)
            except LessonPlan.DoesNotExist:
                return self.error("Lesson plan does not exist")

        lesson_plans = LessonPlan.objects.filter(visible=True).select_related("created_by").order_by("title")
        keyword = request.GET.get("keyword")
        if keyword:
            lesson_plans = lesson_plans.filter(title__icontains=keyword)

        return self.success(self.paginate_data(request, lesson_plans, LessonPlanSerializer))


class LessonPlanProgressAPI(APIView):
    @login_required
    def get(self, request):
        lesson_plan_id = request.GET.get("lesson_plan_id")
        if not lesson_plan_id:
            return self.error("missing lesson_plan_id")
        try:
            lp = LessonPlan.objects.get(id=lesson_plan_id)
        except LessonPlan.DoesNotExist:
            return self.error("Lesson plan does not exist")

        lpps = lp.lessonplanproblem_set.all().select_related("problem")
        problems = []
        solved = 0
        for lpp in lpps:
            pid = lpp.problem_id
            subs = Submission.objects.filter(user_id=request.user.id, problem_id=pid)
            is_solved = subs.filter(result=JudgeStatus.ACCEPTED).exists()
            attempts = subs.count()
            if is_solved:
                solved += 1
            problems.append({"problem_id": pid, "solved": is_solved, "attempts": attempts})
        return self.success({
            "total": len(problems),
            "solved": solved,
            "problems": problems,
        })


class LessonPlanAIGenerateAPI(APIView):
    @login_required
    @transaction.atomic
    def post(self, request):
        topic = (request.data.get("topic") or "").strip()
        if not topic:
            return self.error("请输入主题")

        prompt = (
            f"你是编程教学专家。请为【{topic}】编写一份详细的教案，包含以下部分：\n"
            f"## 学习目标\n## 核心概念讲解\n## 典型例题分析\n## 易错点提醒\n## 小结\n\n"
            f"要求：使用 Markdown 格式，内容详实、条理清晰，适合大学生学习。"
            f"代码示例请用代码块包裹。"
        )
        content = ""
        try:
            for chunk in ask_ai_stream(prompt):
                content += chunk
        except Exception as e:
            return self.error(f"AI 生成失败：{e}")

        if not content.strip():
            return self.error("AI 未返回内容")

        lp = LessonPlan.objects.create(
            title=f"AI 教案 · {topic}",
            description=f"AI 生成的【{topic}】教案",
            content=content,
            created_by=request.user,
            visible=True,
        )

        # 按主题匹配相关题目并挂载，使进度追踪有意义
        related = Problem.objects.filter(is_public=True, visible=True).filter(
            tags__name__icontains=topic
        ).distinct().order_by("accepted_number")[:6]
        if not related.exists():
            related = Problem.objects.filter(
                is_public=True, visible=True, title__icontains=topic
            ).distinct().order_by("accepted_number")[:6]
        for idx, p in enumerate(related):
            LessonPlanProblem.objects.create(lesson_plan=lp, problem=p, order=idx)

        return self.success({
            "id": lp.id,
            "title": lp.title,
            "content": content,
            "problems_count": lp.lessonplanproblem_set.count(),
        })
