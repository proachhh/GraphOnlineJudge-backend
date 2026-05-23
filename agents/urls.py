from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.agent_chat, name='agent_chat'),
    path('profile/init/', views.profile_init, name='profile_init'),
]
