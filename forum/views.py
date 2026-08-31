from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.paginator import Paginator
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.utils import timezone
from account.models import User
from .models import Category, Post, Comment, PostLike, PostBookmark, CommentLike, Report, UserMute
from .serializers import CategorySerializer, PostListSerializer, PostDetailSerializer, CommentSerializer
from utils.api import validate_serializer
from account.decorators import login_required
from .moderation import moderate_content
import os, uuid


class CategoryListAPI(APIView):
    def get(self, request):
        categories = Category.objects.all()
        serializer = CategorySerializer(categories, many=True)
        return Response({"data": serializer.data})


class PostListAPI(APIView):
    def get(self, request):
        category_id = request.GET.get("category_id")
        my_posts = request.GET.get("mine") == "1"
        bookmarked = request.GET.get("bookmarked") == "1"
        page = int(request.GET.get("page", 1))
        limit = int(request.GET.get("limit", 20))
        qs = Post.objects.select_related("author", "category").all()
        if my_posts and request.user.is_authenticated:
            qs = qs.filter(author=request.user)
        elif bookmarked and request.user.is_authenticated:
            qs = qs.filter(bookmarks__user=request.user)
        elif category_id:
            qs = qs.filter(category_id=category_id)
        qs = qs.order_by("-is_pinned", "-create_time")
        paginator = Paginator(qs, limit)
        page_obj = paginator.get_page(page)
        serializer = PostListSerializer(page_obj, many=True, context={"request": request})
        return Response({
            "data": {
                "results": serializer.data,
                "total": paginator.count,
                "page": page,
                "limit": limit,
            }
        })

    @login_required
    def post(self, request):
        # 检查禁言
        try:
            m = request.user.mute_record
            if m.muted_until:
                muted_utc = m.muted_until
                if timezone.is_naive(muted_utc):
                    muted_utc = timezone.make_aware(muted_utc, timezone.utc)
                if muted_utc > timezone.now():
                    return Response({"error": f"你已被禁言：{m.reason}"}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": f"你已被禁言：{m.reason}"}, status=status.HTTP_400_BAD_REQUEST)
            m.delete()
        except UserMute.DoesNotExist:
            pass
        data = request.data
        title = (data.get("title") or "").strip()
        content = (data.get("content") or "").strip()
        category_id = (data.get("category_id") or "").strip()

        if not title or not content or not category_id:
            return Response({"error": "标题、内容和版块不能为空"}, status=status.HTTP_400_BAD_REQUEST)
        if len(title) > 256:
            return Response({"error": "标题过长"}, status=status.HTTP_400_BAD_REQUEST)

        ok, word = moderate_content(title, content, request.user)
        if not ok:
            return Response({"error": f"内容包含违规词语，请修改后重试"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            category = Category.objects.get(id=category_id)
        except Category.DoesNotExist:
            return Response({"error": "版块不存在"}, status=status.HTTP_404_NOT_FOUND)

        post = Post.objects.create(
            title=title, content=content, category=category,
            author=request.user,
        )

        if post.is_locked:
            return Response({"error": "该版块已锁定"}, status=status.HTTP_403_FORBIDDEN)

        return Response({"data": {"id": post.id}}, status=status.HTTP_201_CREATED)


class PostDetailAPI(APIView):
    def get(self, request, post_id):
        try:
            post = Post.objects.select_related("author", "category").get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)
        post.view_count += 1
        post.save(update_fields=["view_count"])
        serializer = PostDetailSerializer(post)
        return Response({"data": serializer.data})

    @login_required
    def delete(self, request, post_id):
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)
        if post.author != request.user and not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        post.delete()
        return Response({"data": "ok"})

    @login_required
    def put(self, request, post_id):
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)
        if post.author != request.user:
            return Response({"error": "只能编辑自己的帖子"}, status=status.HTTP_403_FORBIDDEN)
        title = request.data.get("title", "").strip()
        content = request.data.get("content", "").strip()
        if not title or not content:
            return Response({"error": "标题和内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)
        ok, word = moderate_content(title, content, request.user)
        if not ok:
            return Response({"error": "内容包含违规词语，请修改后重试"}, status=status.HTTP_400_BAD_REQUEST)
        post.title = title
        post.content = content
        post.save(update_fields=["title", "content", "update_time"])
        serializer = PostDetailSerializer(post)
        return Response({"data": serializer.data})


class CommentAPI(APIView):
    def get(self, request):
        return Response({"data": []})

    @login_required
    def post(self, request):
        try:
            m = request.user.mute_record
            if m.muted_until:
                muted_utc = m.muted_until
                if timezone.is_naive(muted_utc):
                    muted_utc = timezone.make_aware(muted_utc, timezone.utc)
                if muted_utc > timezone.now():
                    return Response({"error": f"你已被禁言：{m.reason}"}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": f"你已被禁言：{m.reason}"}, status=status.HTTP_400_BAD_REQUEST)
            m.delete()
        except UserMute.DoesNotExist:
            pass
        post_id = (request.data.get("post_id") or "").strip()
        content = (request.data.get("content") or "").strip()
        parent_id = (request.data.get("parent_id") or "").strip() or None

        if not content or not post_id:
            return Response({"error": "内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        ok, word = moderate_content("", content, request.user)
        if not ok:
            return Response({"error": "评论包含违规词语，请修改后重试"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)

        if post.is_locked:
            return Response({"error": "帖子已锁定"}, status=status.HTTP_403_FORBIDDEN)

        parent = None
        if parent_id:
            try:
                parent = Comment.objects.get(id=parent_id)
            except Comment.DoesNotExist:
                return Response({"error": "父评论不存在"}, status=status.HTTP_404_NOT_FOUND)

        comment = Comment.objects.create(
            content=content, post=post, author=request.user, parent=parent,
        )
        serializer = CommentSerializer(comment)
        return Response({"data": serializer.data}, status=status.HTTP_201_CREATED)

    @login_required
    def delete(self, request):
        comment_id = request.data.get("comment_id", "").strip()
        try:
            comment = Comment.objects.get(id=comment_id)
        except Comment.DoesNotExist:
            return Response({"error": "评论不存在"}, status=status.HTTP_404_NOT_FOUND)
        if comment.author != request.user and not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        comment.delete()
        return Response({"data": "ok"})


class UploadImageAPI(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        f = request.FILES.get("image")
        if not f:
            return Response({"success": False, "msg": "no file"}, status=status.HTTP_400_BAD_REQUEST)
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            return Response({"success": False, "msg": "invalid type"}, status=status.HTTP_400_BAD_REQUEST)
        name = uuid.uuid4().hex[:12] + ext
        dest = os.path.join(settings.UPLOAD_DIR, "forum")
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, name)
        with open(path, "wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
        # 实际存储位置为 UPLOAD_DIR/forum（/data/public/upload/forum），URL 必须与 Nginx 的 /public -> /data 映射一致
        url = f"/public/upload/forum/{name}"
        return Response({"success": True, "file_path": url})


class LikeAPI(APIView):
    @login_required
    def post(self, request):
        post_id = (request.data.get("post_id") or "").strip()
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)
        _, created = PostLike.objects.get_or_create(user=request.user, post=post)
        if not created:
            PostLike.objects.filter(user=request.user, post=post).delete()
        post.like_count = post.likes.count()
        post.save(update_fields=["like_count"])
        return Response({"data": {"liked": created, "like_count": post.like_count}})


class BookmarkAPI(APIView):
    @login_required
    def post(self, request):
        post_id = (request.data.get("post_id") or "").strip()
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"error": "帖子不存在"}, status=status.HTTP_404_NOT_FOUND)
        _, created = PostBookmark.objects.get_or_create(user=request.user, post=post)
        if not created:
            PostBookmark.objects.filter(user=request.user, post=post).delete()
        post.bookmark_count = post.bookmarks.count()
        post.save(update_fields=["bookmark_count"])
        return Response({"data": {"bookmarked": created, "bookmark_count": post.bookmark_count}})


class CommentListAPI(APIView):
    def get(self, request):
        post_id = request.GET.get("post_id", "").strip()
        page = int(request.GET.get("page", 1))
        limit = int(request.GET.get("limit", 2))
        if not post_id:
            return Response({"error": "missing post_id"}, status=status.HTTP_400_BAD_REQUEST)
        qs = Comment.objects.filter(post_id=post_id, parent__isnull=True).order_by("-like_count", "create_time")
        paginator = Paginator(qs, limit)
        page_obj = paginator.get_page(page)
        serializer = CommentSerializer(page_obj, many=True, context={"request": request})
        return Response({"data": {"results": serializer.data, "total": paginator.count, "page": page, "limit": limit}})


class CommentLikeAPI(APIView):
    @login_required
    def post(self, request):
        comment_id = (request.data.get("comment_id") or "").strip()
        try:
            comment = Comment.objects.get(id=comment_id)
        except Comment.DoesNotExist:
            return Response({"error": "评论不存在"}, status=status.HTTP_404_NOT_FOUND)
        _, created = CommentLike.objects.get_or_create(user=request.user, comment=comment)
        if not created:
            CommentLike.objects.filter(user=request.user, comment=comment).delete()
        comment.like_count = comment.likes.count()
        comment.save(update_fields=["like_count"])
        return Response({"data": {"liked": created, "like_count": comment.like_count}})


class ReportAPI(APIView):
    @login_required
    def post(self, request):
        target_type = (request.data.get("target_type") or "").strip()
        target_id = (request.data.get("target_id") or "").strip()
        reason = (request.data.get("reason") or "other").strip()
        detail = (request.data.get("detail") or "").strip()
        if target_type not in ("post", "comment") or not target_id:
            return Response({"error": "参数错误"}, status=status.HTTP_400_BAD_REQUEST)
        Report.objects.create(
            reporter=request.user,
            target_type=target_type, target_id=target_id,
            reason=reason, detail=detail
        )
        return Response({"data": "ok"})


class AdminReportListAPI(APIView):
    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        status_filter = request.GET.get("status", "")
        page = int(request.GET.get("page", 1))
        limit = int(request.GET.get("limit", 20))
        qs = Report.objects.select_related("reporter").all()
        if status_filter:
            qs = qs.filter(status=status_filter)
        paginator = Paginator(qs, limit)
        page_obj = paginator.get_page(page)
        data = []
        for r in page_obj:
            # 获取被举报内容
            target_content = ''
            if r.target_type == 'post':
                try:
                    target_content = Post.objects.get(id=r.target_id).content[:500]
                except Post.DoesNotExist:
                    target_content = '[已删除]'
            elif r.target_type == 'comment':
                try:
                    target_content = Comment.objects.get(id=r.target_id).content[:500]
                except Comment.DoesNotExist:
                    target_content = '[已删除]'
            data.append({
                "id": r.id, "reporter": r.reporter.username if r.reporter else "",
                "target_type": r.target_type, "target_id": r.target_id,
                "reason": r.reason, "detail": r.detail, "status": r.status,
                "create_time": str(r.create_time),
                "target_content": target_content,
            })
        return Response({"data": {"results": data, "total": paginator.count}})

    def put(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        report_id = (request.data.get("report_id") or "").strip()
        action = (request.data.get("action") or "").strip()  # resolve / dismiss
        note = (request.data.get("note") or "").strip()
        try:
            r = Report.objects.get(id=report_id)
        except Report.DoesNotExist:
            return Response({"error": "举报不存在"}, status=status.HTTP_404_NOT_FOUND)
        r.status = "resolved" if action == "resolve" else "dismissed"
        r.handled_by = request.user
        r.handle_note = note
        r.handle_time = timezone.now()
        r.save()
        return Response({"data": "ok"})


class AdminPostListAPI(APIView):
    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        keyword = request.GET.get("keyword", "").strip()
        page = int(request.GET.get("page", 1))
        limit = int(request.GET.get("limit", 20))
        qs = Post.objects.select_related("author").all()
        if keyword:
            qs = qs.filter(title__icontains=keyword)
        paginator = Paginator(qs, limit)
        page_obj = paginator.get_page(page)
        data = []
        for p in page_obj:
            data.append({
                "id": p.id, "title": p.title, "content": p.content,
                "author": p.author.username if p.author else "",
                "view_count": p.view_count, "comment_count": p.comments.count(),
                "is_locked": p.is_locked,
                "create_time": str(p.create_time),
            })
        return Response({"data": {"results": data, "total": paginator.count}})

    def delete(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        post_ids = request.data.get("post_ids", [])
        if post_ids:
            Post.objects.filter(id__in=post_ids).delete()
        return Response({"data": "ok"})


class MuteStatusAPI(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return Response({"data": {"muted": False}})
        try:
            m = request.user.mute_record
            if m.muted_until:
                muted_utc = m.muted_until
                if timezone.is_naive(muted_utc):
                    muted_utc = timezone.make_aware(muted_utc, timezone.utc)
                now = timezone.now()
                if muted_utc < now:
                    m.delete()
                    return Response({"data": {"muted": False}})
                remaining = muted_utc - now
                hours = int(remaining.total_seconds() / 3600)
                remain = f'{hours}小时' if hours > 0 else f'{int(remaining.total_seconds()/60)}分钟'
                return Response({"data": {"muted": True, "reason": m.reason, "remain": remain}})
            return Response({"data": {"muted": True, "reason": m.reason, "remain": "永久"}})
        except UserMute.DoesNotExist:
            return Response({"data": {"muted": False}})


class AdminMuteAPI(APIView):
    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        keyword = request.GET.get("keyword", "").strip()
        page = int(request.GET.get("page", 1))
        offset = int(request.GET.get("offset", 0))
        limit = int(request.GET.get("limit", 20))
        qs = User.objects.all()
        if keyword:
            qs = qs.filter(username__icontains=keyword)
        paginator = Paginator(qs, limit)
        page_num = page if not offset else (offset // limit) + 1
        page_obj = paginator.get_page(page_num)
        data = []
        now = timezone.now()
        for u in page_obj:
            is_muted = False
            mute_reason = ''
            mute_duration = ''
            mute_time = ''
            try:
                m = u.mute_record
                if m.muted_until:
                    muted_utc = m.muted_until
                    if timezone.is_naive(muted_utc):
                        muted_utc = timezone.make_aware(muted_utc, timezone.utc)
                    if muted_utc < now:
                        m.delete()
                    else:
                        is_muted = True
                        mute_reason = m.reason
                        if m.muted_until:
                            remaining = muted_utc - now
                            hours = int(remaining.total_seconds() / 3600)
                            mute_duration = f'{hours}小时' if hours > 0 else f'{int(remaining.total_seconds()/60)}分钟'
                        else:
                            mute_duration = '永久'
                        mute_time = str(m.create_time)
            except UserMute.DoesNotExist:
                pass
            data.append({
                "id": u.id, "username": u.username,
                "is_muted": is_muted, "mute_reason": mute_reason,
                "mute_duration": mute_duration, "mute_time": mute_time,
            })
        return Response({"data": {"results": data, "total": paginator.count}})

    def post(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        user_id = request.data.get("user_id")
        duration = request.data.get("duration")  # hours, or "forever"
        reason = (request.data.get("reason") or "").strip()
        try:
            u = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "用户不存在"}, status=status.HTTP_404_NOT_FOUND)
        until = None if duration == "forever" else timezone.now() + timezone.timedelta(hours=int(duration))
        UserMute.objects.update_or_create(user=u, defaults={"reason": reason, "muted_by": request.user, "muted_until": until})
        return Response({"data": "ok"})

    def delete(self, request):
        if not request.user.is_authenticated or not request.user.is_admin_role():
            return Response({"error": "无权限"}, status=status.HTTP_403_FORBIDDEN)
        user_id = request.data.get("user_id")
        UserMute.objects.filter(user_id=user_id).delete()
        return Response({"data": "ok"})
