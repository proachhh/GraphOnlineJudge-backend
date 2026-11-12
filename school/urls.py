from django.conf.urls import url

from .views import (ClassAPI, ClassDetailAPI, ClassMemberAPI, ClassDashboardAPI,
                    ExamAPI, ExamEnterAPI, ExamProblemAPI, ExamSubmissionAPI,
                    ExamScoreAPI, ExamScoreExportAPI, ExamMonitorAPI)

urlpatterns = [
    url(r"^class/?$", ClassAPI.as_view()),
    url(r"^class/detail/?$", ClassDetailAPI.as_view()),
    url(r"^class/member/?$", ClassMemberAPI.as_view()),
    url(r"^class/dashboard/?$", ClassDashboardAPI.as_view()),
    url(r"^exam/?$", ExamAPI.as_view()),
    url(r"^exam/enter/?$", ExamEnterAPI.as_view()),
    url(r"^exam/problem/?$", ExamProblemAPI.as_view()),
    url(r"^exam/submit/?$", ExamSubmissionAPI.as_view()),
    url(r"^exam/score/?$", ExamScoreAPI.as_view()),
    url(r"^exam/score/export/?$", ExamScoreExportAPI.as_view()),
    url(r"^exam/monitor/?$", ExamMonitorAPI.as_view()),
]
