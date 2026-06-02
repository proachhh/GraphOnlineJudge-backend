from django.conf.urls import url
from ..views_admin import AdminFeedbackAPI

urlpatterns = [
    url(r"^feedback/?$", AdminFeedbackAPI.as_view(), name="admin_feedback_api"),
]
