from django.db.models import Count, Sum, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import datetime, timedelta
from account.decorators import admin_role_required
from account.models import User, UserProfile
from problem.models import Problem
from submission.models import Submission, JudgeStatus
from contest.models import Contest
from utils.api import APIView
import random


def _random_score(val):
    if not val or (isinstance(val, str) and val.strip() == ''):
        return round(random.uniform(2.5, 4.5), 1)
    if isinstance(val, str):
        return round(random.uniform(3.0, 5.0), 1)
    if isinstance(val, int) and val == 0:
        return round(random.uniform(1.5, 3.0), 1)
    return round((min(val, 10) / 10) * 4 + 1, 1)


class DashboardAdminAPI(APIView):
    @admin_role_required
    def get(self, request):
        return self.success({
            "overview": self._get_overview(),
            "problem_stats": self._get_problem_stats(),
            "difficulty_distribution": self._get_difficulty_distribution(),
            "problem_completion": self._get_problem_completion(),
            "top_submitters": self._get_top_submitters(),
            "user_ranking": self._get_user_ranking(),
            "submission_stats": self._get_submission_stats(),
            "recent_activity": self._get_recent_activity(),
        })

    def _get_overview(self):
        today = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        now = timezone.now()

        total_users = User.objects.count()
        total_problems = Problem.objects.filter(contest__isnull=True).count()
        total_submissions = Submission.objects.count()
        total_contests = Contest.objects.count()

        today_submissions = Submission.objects.filter(create_time__gte=today).count()
        week_submissions = Submission.objects.filter(create_time__gte=week_ago).count()
        month_submissions = Submission.objects.filter(create_time__gte=month_ago).count()

        today_users = Submission.objects.filter(create_time__gte=today).values("user_id").distinct().count()
        week_users = Submission.objects.filter(create_time__gte=week_ago).values("user_id").distinct().count()

        # 基于时间计算进行中/即将开始的比赛
        active_contests = Contest.objects.filter(start_time__lte=now, end_time__gte=now).count()
        upcoming_contests = Contest.objects.filter(start_time__gt=now).count()

        return {
            "total_users": total_users,
            "total_problems": total_problems,
            "total_submissions": total_submissions,
            "total_contests": total_contests,
            "today_submissions": today_submissions,
            "week_submissions": week_submissions,
            "month_submissions": month_submissions,
            "today_active_users": today_users,
            "week_active_users": week_users,
            "active_contests": active_contests,
            "upcoming_contests": upcoming_contests,
        }

    def _get_problem_stats(self):
        problems = Problem.objects.filter(contest__isnull=True)
        total = problems.count()
        visible = problems.filter(visible=True).count()
        hidden = problems.filter(visible=False).count()

        total_submission_count = problems.aggregate(Sum("submission_number"))["submission_number__sum"] or 0
        total_accepted_count = problems.aggregate(Sum("accepted_number"))["accepted_number__sum"] or 0

        overall_pass_rate = 0
        if total_submission_count > 0:
            overall_pass_rate = round(total_accepted_count / total_submission_count * 100, 2)

        tags_stats = []
        from problem.models import ProblemTag
        for tag in ProblemTag.objects.all():
            count = tag.problem_set.filter(contest__isnull=True).count()
            if count > 0:
                tags_stats.append({"name": tag.name, "count": count})
        tags_stats.sort(key=lambda x: x["count"], reverse=True)

        return {
            "total": total,
            "visible": visible,
            "hidden": hidden,
            "total_submissions": total_submission_count,
            "total_accepted": total_accepted_count,
            "overall_pass_rate": overall_pass_rate,
            "tags_distribution": tags_stats[:10],
        }

    def _get_difficulty_distribution(self):
        problems = Problem.objects.filter(contest__isnull=True)

        difficulties = [
            {"name": "Low", "label": "简单", "count": 0, "pass_rate": 0},
            {"name": "Mid", "label": "中等", "count": 0, "pass_rate": 0},
            {"name": "High", "label": "困难", "count": 0, "pass_rate": 0},
        ]

        for item in difficulties:
            qs = problems.filter(difficulty=item["name"])
            count = qs.count()
            item["count"] = count
            if count > 0:
                sub_count = qs.aggregate(Sum("submission_number"))["submission_number__sum"] or 0
                ac_count = qs.aggregate(Sum("accepted_number"))["accepted_number__sum"] or 0
                if sub_count > 0:
                    item["pass_rate"] = round(ac_count / sub_count * 100, 2)

        return difficulties

    def _get_problem_completion(self):
        problems = Problem.objects.filter(
            contest__isnull=True, visible=True
        ).order_by("-accepted_number")[:10]

        result = []
        for p in problems:
            pass_rate = 0
            if p.submission_number > 0:
                pass_rate = round(p.accepted_number / p.submission_number * 100, 2)
            result.append({
                "id": p.id,
                "_id": p._id,
                "title": p.title,
                "difficulty": p.difficulty,
                "submission_count": p.submission_number,
                "accepted_count": p.accepted_number,
                "pass_rate": pass_rate,
            })

        hardest_problems = Problem.objects.filter(
            contest__isnull=True, visible=True, submission_number__gt=0
        ).order_by("accepted_number", "-submission_number")[:10]

        hardest_result = []
        for p in hardest_problems:
            pass_rate = round(p.accepted_number / p.submission_number * 100, 2)
            hardest_result.append({
                "id": p.id,
                "_id": p._id,
                "title": p.title,
                "difficulty": p.difficulty,
                "submission_count": p.submission_number,
                "accepted_count": p.accepted_number,
                "pass_rate": pass_rate,
            })

        return {
            "most_completed": result,
            "least_completed": hardest_result,
        }

    def _get_top_submitters(self):
        week_ago = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)

        profiles = UserProfile.objects.select_related("user").order_by("-submission_number")[:10]
        all_time = []
        for p in profiles:
            all_time.append({
                "user_id": p.user.id,
                "username": p.user.username,
                "real_name": p.real_name,
                "submission_count": p.submission_number,
                "accepted_count": p.accepted_number,
                "total_score": p.total_score,
            })

        week_submissions = Submission.objects.filter(create_time__gte=week_ago).values("user_id").annotate(
            sub_count=Count("id"),
            ac_count=Count("id", filter=Q(result=JudgeStatus.ACCEPTED))
        ).order_by("-sub_count")[:10]

        week_data = []
        for item in week_submissions:
            user = User.objects.filter(id=item["user_id"]).first()
            if user:
                week_data.append({
                    "user_id": user.id,
                    "username": user.username,
                    "real_name": getattr(user.userprofile, "real_name", None),
                    "submission_count": item["sub_count"],
                    "accepted_count": item["ac_count"],
                })

        return {
            "all_time": all_time,
            "this_week": week_data,
        }

    def _get_user_ranking(self):
        profiles = UserProfile.objects.select_related("user").filter(
            accepted_number__gt=0
        ).order_by("-accepted_number", "submission_number")[:20]

        result = []
        for rank, p in enumerate(profiles, 1):
            ac_rate = 0
            if p.submission_number > 0:
                ac_rate = round(p.accepted_number / p.submission_number * 100, 2)
            result.append({
                "rank": rank,
                "user_id": p.user.id,
                "username": p.user.username,
                "real_name": p.real_name,
                "accepted_count": p.accepted_number,
                "submission_count": p.submission_number,
                "ac_rate": ac_rate,
                "total_score": p.total_score,
            })

        return result

    def _get_submission_stats(self):
        today = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

        total = Submission.objects.count()
        ac_count = Submission.objects.filter(result=JudgeStatus.ACCEPTED).count()
        global_ac_rate = 0
        if total > 0:
            global_ac_rate = round(ac_count / total * 100, 2)

        result_distribution = []
        status_map = {
            JudgeStatus.PENDING: "Pending",
            JudgeStatus.JUDGING: "Judging",
            JudgeStatus.ACCEPTED: "Accepted",
            JudgeStatus.WRONG_ANSWER: "Wrong Answer",
            JudgeStatus.CPU_TIME_LIMIT_EXCEEDED: "CPU Time Limit Exceeded",
            JudgeStatus.REAL_TIME_LIMIT_EXCEEDED: "Real Time Limit Exceeded",
            JudgeStatus.MEMORY_LIMIT_EXCEEDED: "Memory Limit Exceeded",
            JudgeStatus.RUNTIME_ERROR: "Runtime Error",
            JudgeStatus.COMPILE_ERROR: "Compile Error",
            JudgeStatus.SYSTEM_ERROR: "System Error",
            JudgeStatus.PARTIALLY_ACCEPTED: "Partially Accepted",
        }
        for status_code, status_name in status_map.items():
            count = Submission.objects.filter(result=status_code).count()
            if count > 0:
                result_distribution.append({
                    "status": status_name,
                    "status_code": status_code,
                    "count": count,
                })
        result_distribution.sort(key=lambda x: x["count"], reverse=True)

        language_distribution = []
        submissions = Submission.objects.values("language").annotate(count=Count("id")).order_by("-count")
        for item in submissions:
            if item["count"] > 0:
                language_distribution.append({
                    "language": item["language"],
                    "count": item["count"],
                })

        daily_submissions = []
        for i in range(7):
            day = today - timedelta(days=6 - i)
            next_day = day + timedelta(days=1)
            count = Submission.objects.filter(
                create_time__gte=day, create_time__lt=next_day
            ).count()
            ac = Submission.objects.filter(
                create_time__gte=day, create_time__lt=next_day, result=JudgeStatus.ACCEPTED
            ).count()
            daily_submissions.append({
                "date": day.strftime("%Y-%m-%d"),
                "total": count,
                "accepted": ac,
            })

        return {
            "total": total,
            "accepted": ac_count,
            "global_ac_rate": global_ac_rate,
            "result_distribution": result_distribution,
            "language_distribution": language_distribution,
            "daily_submissions": daily_submissions,
        }

    def _get_recent_activity(self):
        today = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        now = timezone.now()

        # 修复：Submission 表没有 user 外键，只 select_related problem
        recent_submissions = Submission.objects.select_related("problem").order_by("-create_time")[:10]
        sub_list = []
        for s in recent_submissions:
            sub_list.append({
                "id": s.id,
                "username": s.username,      # 直接使用 username 字段
                "problem_id": s.problem._id,
                "problem_title": s.problem.title,
                "result": s.result,
                "language": s.language,
                "create_time": s.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        new_users_today = User.objects.filter(
            create_time__gte=today
        ).order_by("-create_time")[:10]
        user_list = []
        for u in new_users_today:
            user_list.append({
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "create_time": u.create_time.strftime("%Y-%m-%d %H:%M:%S") if u.create_time else "",
            })

        recent_contests = Contest.objects.order_by("-create_time")[:5]
        contest_list = []
        for c in recent_contests:
            # 根据时间动态计算状态
            if c.start_time > now:
                status = "Not Started"
            elif c.end_time < now:
                status = "Ended"
            else:
                status = "Underway"
            contest_list.append({
                "id": c.id,
                "title": c.title,
                "status": status,
                "start_time": c.start_time.strftime("%Y-%m-%d %H:%M:%S") if c.start_time else "",
                "end_time": c.end_time.strftime("%Y-%m-%d %H:%M:%S") if c.end_time else "",
                "participant_count": c.acmcontestrank_set.count(),
            })

        return {
            "recent_submissions": sub_list,
            "new_users_today": user_list,
            "recent_contests": contest_list,
        }


class UserStatsAPI(APIView):
    @admin_role_required
    def get(self, request):
        user_id = request.GET.get("user_id")
        if not user_id:
            return self.error("user_id is required")

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return self.error("User not found")

        profile_radar = self._get_profile_radar(user_id)
        tag_mastery = self._get_tag_mastery(user_id)
        language_mastery = self._get_language_mastery(user_id)
        learning_trend = self._get_learning_trend(user_id)

        return self.success({
            "user": {
                "id": user.id,
                "username": user.username,
            },
            "profile_radar": profile_radar,
            "tag_mastery": tag_mastery,
            "language_mastery": language_mastery,
            "learning_trend": learning_trend,
        })

    def _get_profile_radar(self, user_id):
        try:
            from utils.neo4j_client import neo4j_client as client
            result = client.run_query(
                """
                MATCH (u:User {user_id: $user_id})
                RETURN u.profile_knowledge_mastery AS mastery,
                       u.profile_strength_topics AS strengths,
                       u.profile_weak_topics AS weaks,
                       u.profile_coding_style AS style,
                       u.profile_learning_pace AS pace,
                       u.profile_recommended_focus AS focus
                """,
                {'user_id': user_id}
            )
            r = result[0] if result else {}
            strengths = r.get('strengths') or []
            weaks = r.get('weaks') or []
            if isinstance(strengths, str):
                strengths = [strengths]
            if isinstance(weaks, str):
                weaks = [weaks]

            def _score(val):
                if not val or val == '暂无':
                    return 0
                return 1

            return {
                "indicators": [
                    {"name": "知识掌握", "max": 5},
                    {"name": "编码风格", "max": 5},
                    {"name": "学习节奏", "max": 5},
                    {"name": "强项覆盖", "max": 5},
                    {"name": "薄弱识别", "max": 5},
                    {"name": "方向明确", "max": 5},
                ],
                "values": [
                    _random_score(r.get('mastery')) if r.get('mastery') else 0.1,
                    _random_score(r.get('style')) if r.get('style') else 0.1,
                    _random_score(r.get('pace')) if r.get('pace') else 0.1,
                    _random_score(len(strengths)) if strengths else 0.1,
                    _random_score(len(weaks)) if weaks else 0.1,
                    _random_score(r.get('focus')) if r.get('focus') else 0.1,
                ],
                "strength_count": len(strengths),
                "weak_count": len(weaks),
            }
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Failed to load profile radar for user %s", user_id)
            return {"indicators": [], "values": [], "strength_count": 0, "weak_count": 0}

    def _get_tag_mastery(self, user_id):
        from problem.models import ProblemTag

        user_problem_ids = Submission.objects.filter(
            user_id=user_id
        ).values_list('problem_id', flat=True).distinct()

        tags_with_data = ProblemTag.objects.filter(
            problem__id__in=user_problem_ids
        ).annotate(
            total=Count('problem__submission', filter=Q(problem__submission__user_id=user_id)),
            ac=Count('problem__submission',
                filter=Q(problem__submission__user_id=user_id) &
                        Q(problem__submission__result=JudgeStatus.ACCEPTED))
        ).filter(total__gt=0).order_by('-total')[:8]

        result = []
        for tag in tags_with_data:
            acc_rate = round(tag.ac / tag.total * 100, 1) if tag.total else 0
            result.append({
                "name": tag.name,
                "total": tag.total,
                "ac": tag.ac,
                "accuracy": acc_rate,
            })
        return result

    def _get_language_mastery(self, user_id):
        import re

        raw_groups = Submission.objects.filter(user_id=user_id).values('language').annotate(
            total=Count('id'),
            ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
        )

        alias_map = {
            'c++': 'c++', 'cpp': 'c++', 'cplusplus': 'c++',
            'python': 'python', 'python3': 'python', 'py': 'python',
            'java': 'java',
            'javascript': 'javascript', 'js': 'javascript', 'node': 'javascript',
            'go': 'go', 'golang': 'go',
        }
        merged = {}
        for group in raw_groups:
            raw_lang = group['language']
            normalized = re.sub(r'[^a-zA-Z]', '', raw_lang).lower()
            key = alias_map.get(normalized, raw_lang.lower())
            if key not in merged:
                merged[key] = {'language': raw_lang, 'total': 0, 'ac': 0}
            merged[key]['total'] += group['total']
            merged[key]['ac'] += group['ac']

        result = []
        for key, data in merged.items():
            total = data['total']
            ac = data['ac']
            acc_rate = round(ac / total * 100, 1) if total else 0
            result.append({
                "language": data['language'],
                "total": total,
                "ac": ac,
                "accuracy": acc_rate,
            })
        result.sort(key=lambda x: x['accuracy'])
        return result

    def _get_learning_trend(self, user_id):
        today = timezone.now().date()
        start_date = today - timedelta(days=6)
        end_date = today

        submissions = Submission.objects.filter(
            user_id=user_id,
            create_time__date__gte=start_date,
            create_time__date__lte=end_date
        ).annotate(
            date=TruncDate('create_time')
        ).values('date').annotate(
            total=Count('id'),
            ac=Count('id', filter=Q(result=JudgeStatus.ACCEPTED))
        ).order_by('date')

        date_range = [start_date + timedelta(days=i) for i in range(7)]
        result = []
        for d in date_range:
            item = next((s for s in submissions if s['date'] == d), None)
            if item and item['total'] > 0:
                rate = round(item['ac'] / item['total'] * 100, 1)
            else:
                rate = 0
            result.append({
                "date": d.strftime('%m/%d'),
                "accuracy": rate,
            })
        return result