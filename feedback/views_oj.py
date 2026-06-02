from account.decorators import login_required
from utils.api import APIView, validate_serializer
from .models import Feedback
from .serializers import CreateFeedbackSerializer, FeedbackSerializer


class FeedbackAPI(APIView):
    @login_required
    @validate_serializer(CreateFeedbackSerializer)
    def post(self, request):
        data = request.data
        fb = Feedback.objects.create(
            user_id=request.user.id,
            username=request.user.username,
            title=data["title"],
            content=data.get("content", "")
        )
        return self.success(FeedbackSerializer(fb).data)
