from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.agent_chat, name='agent_chat'),
    path('profile/init/', views.profile_init, name='profile_init'),
    path('recommend/', views.agent_recommend, name='agent_recommend'),
    path('immersion/', views.agent_immersion, name='agent_immersion'),
    path('learning-path/', views.agent_learning_path, name='agent_learning_path'),
]
