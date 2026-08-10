from django.conf.urls import url
from ..views.oj import LessonPlanAPI, LessonPlanProgressAPI, LessonPlanAIGenerateAPI

urlpatterns = [
    url(r"^lesson_plan/progress/?$", LessonPlanProgressAPI.as_view(), name="lesson_plan_progress"),
    url(r"^lesson_plan/ai_generate/?$", LessonPlanAIGenerateAPI.as_view(), name="lesson_plan_ai_generate"),
    url(r"^lesson_plan/?$", LessonPlanAPI.as_view(), name="lesson_plan_api"),
]
