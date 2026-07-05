from rest_framework import serializers
from .models import Category, Post, Comment


class CategorySerializer(serializers.ModelSerializer):
    post_count = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ["id", "name", "description", "icon", "sort_order", "post_count", "create_time"]

    def get_post_count(self, obj):
        return obj.posts.count()


class UserBriefSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    username = serializers.CharField()
    avatar = serializers.SerializerMethodField()

    def get_avatar(self, obj):
        try:
            return obj.userprofile.avatar or ''
        except Exception:
            return ''


class CommentSerializer(serializers.ModelSerializer):
    author = UserBriefSerializer(read_only=True)
    replies = serializers.SerializerMethodField()
    is_liked = serializers.SerializerMethodField()

    class Meta:
        model = Comment
        fields = ["id", "content", "author", "parent_id", "like_count", "is_liked", "replies", "create_time"]

    def get_replies(self, obj):
        if obj.replies.exists():
            return CommentSerializer(obj.replies.all().order_by('-like_count', 'create_time'), many=True).data
        return []

    def get_is_liked(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            return obj.likes.filter(user=request.user).exists()
        return False


class PostListSerializer(serializers.ModelSerializer):
    author = UserBriefSerializer(read_only=True)
    comment_count = serializers.SerializerMethodField()
    is_liked = serializers.SerializerMethodField()
    is_bookmarked = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id", "title", "content", "author",
            "view_count", "like_count", "bookmark_count", "comment_count",
            "is_liked", "is_bookmarked",
            "is_pinned", "is_locked",
            "create_time", "update_time",
        ]

    def get_comment_count(self, obj):
        return obj.comments.count()

    def get_is_liked(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            return obj.likes.filter(user=request.user).exists()
        return False

    def get_is_bookmarked(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            return obj.bookmarks.filter(user=request.user).exists()
        return False


class PostDetailSerializer(serializers.ModelSerializer):
    author = UserBriefSerializer(read_only=True)
    comment_count = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id", "title", "content", "author",
            "view_count", "like_count", "bookmark_count", "comment_count",
            "is_pinned", "is_locked",
            "create_time", "update_time",
        ]

    def get_comment_count(self, obj):
        return obj.comments.count()
