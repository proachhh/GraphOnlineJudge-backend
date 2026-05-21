from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.agent_chat, name='agent_chat'),
]
