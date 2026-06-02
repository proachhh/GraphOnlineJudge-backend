from django.conf.urls import url
from ..views_oj import FeedbackAPI

urlpatterns = [
    url(r"^feedback/?$", FeedbackAPI.as_view(), name="feedback_api"),
]
