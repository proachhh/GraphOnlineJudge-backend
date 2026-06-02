from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='spark_chat'),
    path('chat/stream/', views.chat_stream, name='spark_chat_stream'),
    path('analyze-error/', views.analyze_error, name='analyze_error'),
    path('problem-hint/', views.problem_hint, name='problem_hint'),
    path('learning-advice/', views.learning_advice, name='learning_advice'),
    path('code-review/', views.code_review, name='code_review'),
    path('topic-summary/', views.topic_summary, name='topic_summary'),
]