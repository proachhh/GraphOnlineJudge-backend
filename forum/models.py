from django.db import models
from utils.models import RichTextField
from utils.shortcuts import rand_str


class Category(models.Model):
    id = models.CharField(max_length=32, default=rand_str, primary_key=True, db_index=True)
    name = models.CharField(max_length=64, unique=True)
    description = models.CharField(max_length=256, null=True, blank=True)
    icon = models.CharField(max_length=32, default="ios-chatbubbles")
    sort_order = models.IntegerField(default=0)
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_category"
        ordering = ("sort_order", "-create_time")

    def __str__(self):
        return self.name


class Post(models.Model):
    id = models.CharField(max_length=32, default=rand_str, primary_key=True, db_index=True)
    title = models.CharField(max_length=256)
    content = RichTextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name="posts")
    author = models.ForeignKey("account.User", on_delete=models.CASCADE, related_name="forum_posts")
    view_count = models.IntegerField(default=0)
    like_count = models.IntegerField(default=0)
    bookmark_count = models.IntegerField(default=0)
    is_pinned = models.BooleanField(default=False)
    is_locked = models.BooleanField(default=False)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "forum_post"
        ordering = ("-is_pinned", "-create_time")

    def __str__(self):
        return self.title


class Comment(models.Model):
    id = models.CharField(max_length=32, default=rand_str, primary_key=True, db_index=True)
    content = models.TextField()
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey("account.User", on_delete=models.CASCADE, related_name="forum_comments")
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True, related_name="replies")
    like_count = models.IntegerField(default=0)
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_comment"
        ordering = ("create_time",)

    def __str__(self):
        return self.content[:64]


class PostLike(models.Model):
    user = models.ForeignKey("account.User", on_delete=models.CASCADE)
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name="likes")
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_post_like"
        unique_together = ("user", "post")


class PostBookmark(models.Model):
    user = models.ForeignKey("account.User", on_delete=models.CASCADE)
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name="bookmarks")
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_post_bookmark"
        unique_together = ("user", "post")


class CommentLike(models.Model):
    user = models.ForeignKey("account.User", on_delete=models.CASCADE)
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE, related_name="likes")
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_comment_like"
        unique_together = ("user", "comment")


class Report(models.Model):
    REPORT_TYPES = [
        ("spam", "垃圾信息"),
        ("attack", "人身攻击"),
        ("ad", "广告推广"),
        ("illegal", "违规内容"),
        ("other", "其他"),
    ]
    REPORT_STATUS = [
        ("pending", "待处理"),
        ("resolved", "已处理"),
        ("dismissed", "已驳回"),
    ]
    id = models.CharField(max_length=32, default=rand_str, primary_key=True, db_index=True)
    reporter = models.ForeignKey("account.User", on_delete=models.CASCADE, related_name="reports_made")
    target_type = models.CharField(max_length=16)  # 'post' or 'comment'
    target_id = models.CharField(max_length=32)     # post.id or comment.id
    reason = models.CharField(max_length=32, choices=REPORT_TYPES, default="other")
    detail = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=REPORT_STATUS, default="pending")
    handled_by = models.ForeignKey("account.User", on_delete=models.SET_NULL, null=True, blank=True, related_name="reports_handled")
    handle_note = models.TextField(null=True, blank=True)
    create_time = models.DateTimeField(auto_now_add=True)
    handle_time = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "forum_report"
        ordering = ("-create_time",)


class UserMute(models.Model):
    user = models.OneToOneField("account.User", on_delete=models.CASCADE, related_name="mute_record")
    reason = models.TextField()
    muted_by = models.ForeignKey("account.User", on_delete=models.SET_NULL, null=True, related_name="mutes_given")
    muted_until = models.DateTimeField(null=True, blank=True)  # null = 永久
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forum_user_mute"
