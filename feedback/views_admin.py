from account.decorators import super_admin_required
from utils.api import APIView
from .models import Feedback
from .serializers import AdminFeedbackSerializer, FeedbackSerializer


class AdminFeedbackAPI(APIView):
    @super_admin_required
    def get(self, request):
        feedbacks = Feedback.objects.all().order_by("-create_time")
        resolved = request.GET.get("resolved")
        if resolved is not None:
            feedbacks = feedbacks.filter(resolved=resolved == "true")
        data = self.paginate_data(request, feedbacks, FeedbackSerializer)
        return self.success(data)

    @super_admin_required
    def put(self, request):
        feedback_id = request.data.get("feedback_id")
        is_resolved = request.data.get("resolved")
        admin_note = request.data.get("admin_note")
        if not feedback_id:
            return self.error("feedback_id is required")
        try:
            fb = Feedback.objects.get(id=feedback_id)
        except Feedback.DoesNotExist:
            return self.error("Feedback not found")
        updated = []
        if is_resolved is not None:
            fb.resolved = is_resolved
            updated.append("resolved")
        if admin_note is not None:
            fb.admin_note = admin_note
            updated.append("admin_note")
        if updated:
            fb.save(update_fields=updated)
        return self.success(FeedbackSerializer(fb).data)

    @super_admin_required
    def delete(self, request):
        feedback_id = request.GET.get("feedback_id")
        if not feedback_id:
            return self.error("feedback_id is required")
        Feedback.objects.filter(id=feedback_id).delete()
        return self.success()
