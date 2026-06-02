from utils.api import serializers
from .models import Feedback


class CreateFeedbackSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=128)
    content = serializers.CharField(max_length=10000, allow_blank=True)


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = "__all__"


class AdminFeedbackSerializer(serializers.Serializer):
    feedback_id = serializers.CharField(max_length=32)
    resolved = serializers.BooleanField(required=False)
    admin_note = serializers.CharField(max_length=5000, required=False, allow_blank=True, allow_null=True)
