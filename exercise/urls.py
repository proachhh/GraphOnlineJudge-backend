from django.urls import re_path

from . import views

urlpatterns = [
    # Student - ExerciseSet
    re_path(r'^topics/?$', views.exercise_topics, name='exercise_topics'),
    re_path(r'^sets/?$', views.exercise_set_list, name='exercise_set_list'),
    re_path(r'^sets/(?P<set_id>\d+)/?$', views.exercise_set_detail, name='exercise_set_detail'),
    re_path(r'^sets/(?P<set_id>\d+)/start/?$', views.exercise_start, name='exercise_start'),
    re_path(r'^sets/(?P<set_id>\d+)/submit/?$', views.exercise_submit, name='exercise_submit'),
    re_path(r'^submissions/(?P<sub_id>\d+)/report/?$', views.exercise_report, name='exercise_report'),
    re_path(r'^submissions/(?P<sub_id>\d+)/grade-stream/?$', views.exercise_grade_stream, name='exercise_grade_stream'),
    re_path(r'^sets/(?P<set_id>\d+)/ranking/?$', views.exercise_ranking, name='exercise_ranking'),

    # Student - BossExam
    re_path(r'^boss/?$', views.boss_exam_list, name='boss_exam_list'),
    re_path(r'^boss/(?P<exam_id>\d+)/?$', views.boss_exam_detail, name='boss_exam_detail'),
    re_path(r'^boss/(?P<exam_id>\d+)/start/?$', views.boss_exam_start, name='boss_exam_start'),
    re_path(r'^boss/(?P<exam_id>\d+)/submit/?$', views.boss_exam_submit, name='boss_exam_submit'),
    re_path(r'^boss/submissions/(?P<sub_id>\d+)/report/?$', views.boss_exam_report, name='boss_exam_report'),
    re_path(r'^boss/submissions/(?P<sub_id>\d+)/grade-stream/?$', views.boss_grade_stream, name='boss_grade_stream'),

    # Teacher
    re_path(r'^teacher/sets/?$', views.teacher_exercise_set_list, name='teacher_exercise_set_list'),
    re_path(r'^teacher/sets/create/?$', views.teacher_exercise_set_create, name='teacher_exercise_set_create'),
    re_path(r'^teacher/sets/(?P<set_id>\d+)/update/?$', views.teacher_exercise_set_update, name='teacher_exercise_set_update'),
    re_path(r'^teacher/sets/(?P<set_id>\d+)/delete/?$', views.teacher_exercise_set_delete, name='teacher_exercise_set_delete'),
    re_path(r'^teacher/sets/(?P<set_id>\d+)/publish/?$', views.teacher_exercise_set_publish, name='teacher_exercise_set_publish'),
    re_path(r'^teacher/sets/(?P<set_id>\d+)/submissions/?$', views.teacher_exercise_submissions, name='teacher_exercise_submissions'),
    re_path(r'^teacher/ai/generate-questions/?$', views.teacher_ai_generate_questions, name='teacher_ai_generate_questions'),
    re_path(r'^teacher/boss/?$', views.teacher_boss_exam_list, name='teacher_boss_exam_list'),
    re_path(r'^teacher/boss/create/?$', views.teacher_boss_exam_create, name='teacher_boss_exam_create'),
    re_path(r'^teacher/boss/(?P<exam_id>\d+)/update/?$', views.teacher_boss_exam_update, name='teacher_boss_exam_update'),
    re_path(r'^teacher/boss/(?P<exam_id>\d+)/delete/?$', views.teacher_boss_exam_delete, name='teacher_boss_exam_delete'),
    re_path(r'^teacher/boss/(?P<exam_id>\d+)/publish/?$', views.teacher_boss_exam_publish, name='teacher_boss_exam_publish'),
    re_path(r'^teacher/students/?$', views.teacher_student_list, name='teacher_student_list'),
    re_path(r'^teacher/students/(?P<user_id>\d+)/report/?$', views.teacher_student_report, name='teacher_student_report'),
    re_path(r'^teacher/students/(?P<user_id>\d+)/ai-analysis/?$', views.teacher_student_ai_analysis, name='teacher_student_ai_analysis'),
    re_path(r'^teacher/topics/?$', views.teacher_topic_list, name='teacher_topic_list'),
    re_path(r'^teacher/stats/?$', views.teacher_stats, name='teacher_stats'),
    re_path(r'^teacher/code-check/?$', views.teacher_code_check, name='teacher_code_check'),
]
