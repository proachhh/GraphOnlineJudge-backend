from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='spark_chat'),
    path('chat/stream/', views.chat_stream, name='spark_chat_stream'),
    path('analyze-error/', views.analyze_error, name='analyze_error'),
    path('analyze-error/stream/', views.analyze_error_stream, name='analyze_error_stream'),
    path('problem-hint/', views.problem_hint, name='problem_hint'),
    path('problem-hint/stream/', views.problem_hint_stream, name='problem_hint_stream'),
    path('learning-advice/', views.learning_advice, name='learning_advice'),
    path('learning-advice/stream/', views.learning_advice_stream, name='learning_advice_stream'),
    path('code-review/', views.code_review, name='code_review'),
    path('code-review/stream/', views.code_review_stream, name='code_review_stream'),
    path('code-review-structured/', views.code_review_structured, name='code_review_structured'),
    path('visualize-code/', views.visualize_code, name='visualize_code'),
    path('topic-summary/', views.topic_summary, name='topic_summary'),
]