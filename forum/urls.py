from django.conf.urls import url
from .views import (CategoryListAPI, PostListAPI, PostDetailAPI, CommentAPI, CommentListAPI,
                    UploadImageAPI, LikeAPI, BookmarkAPI, CommentLikeAPI,
                    ReportAPI, AdminReportListAPI, AdminPostListAPI, AdminMuteAPI,
                    MuteStatusAPI)

urlpatterns = [
    url(r"^forum/categories/?$", CategoryListAPI.as_view(), name="forum_categories"),
    url(r"^forum/posts/?$", PostListAPI.as_view(), name="forum_posts"),
    url(r"^forum/post/(?P<post_id>[a-zA-Z0-9]+)/?$", PostDetailAPI.as_view(), name="forum_post_detail"),
    url(r"^forum/comments/?$", CommentAPI.as_view(), name="forum_comments"),
    url(r"^forum/comment_list/?$", CommentListAPI.as_view(), name="forum_comment_list"),
    url(r"^forum/upload_image/?$", UploadImageAPI.as_view(), name="forum_upload_image"),
    url(r"^forum/like/?$", LikeAPI.as_view(), name="forum_like"),
    url(r"^forum/bookmark/?$", BookmarkAPI.as_view(), name="forum_bookmark"),
    url(r"^forum/comment_like/?$", CommentLikeAPI.as_view(), name="forum_comment_like"),
    url(r"^forum/report/?$", ReportAPI.as_view(), name="forum_report"),
    url(r"^forum/mute_status/?$", MuteStatusAPI.as_view(), name="forum_mute_status"),
    # admin
    url(r"^admin/forum/reports/?$", AdminReportListAPI.as_view(), name="admin_forum_reports"),
    url(r"^admin/forum/posts/?$", AdminPostListAPI.as_view(), name="admin_forum_posts"),
    url(r"^admin/forum/mute/?$", AdminMuteAPI.as_view(), name="admin_forum_mute"),
]
