import ipaddress
import random
from django.db.models import Q, Count
from django.http import HttpResponse
from django.utils.timezone import now

from account.decorators import login_required
from account.models import User, UserProfile
from problem.models import Problem
from problem.serializers import ProblemSerializer
from submission.models import Submission
from judge.tasks import judge_task
from utils.api import APIView, validate_serializer

from .models import (Class, ClassMember, Exam, ExamClass, ExamProblem,
                     ExamParticipation, ExamMonitorLog)
from .serializers import (CreateClassSerializer, EditClassSerializer, JoinClassSerializer,
                          MemberActionSerializer, CreateExamSerializer, EditExamSerializer,
                          ExamSubmitSerializer, MonitorReportSerializer)

import datetime

import pytz

CN_TZ = pytz.timezone("Asia/Shanghai")


def _is_teacher(user):
    return user.is_authenticated and user.is_teacher_or_admin()


def _get_class(id):
    try:
        return Class.objects.get(id=id, status="active")
    except Class.DoesNotExist:
        return None


def _is_class_manager(user, cls):
    """班级创建者 或 管理员"""
    return _is_teacher(user) and (cls.created_by_id == user.id or user.is_admin_role())


def _is_class_assistant(user, cls):
    """助教：也可以管理成员/考试"""
    if _is_class_manager(user, cls):
        return True
    return ClassMember.objects.filter(class_field=cls, user=user,
                                      role="assistant", status="active").exists()


def _is_class_member(user, cls):
    return ClassMember.objects.filter(class_field=cls, user=user, status="active").exists()


def _class_ids_of_user(user):
    """用户所在(或创建的)班级 id 集合"""
    ids = set(ClassMember.objects.filter(user=user, status="active").values_list("class_field_id", flat=True))
    if _is_teacher(user):
        ids |= set(Class.objects.filter(created_by=user, status="active").values_list("id", flat=True))
    return ids


def _exam_problem_queryset(exam, paper_index):
    """学生在自己套卷中能看到的题目（公共卷 + 自己卷）"""
    if exam.paper_count > 1:
        return exam.exam_problems.filter(Q(paper_index=0) | Q(paper_index=paper_index))
    return exam.exam_problems.all()


def _fmt_dt(dt):
    """datetime 转字符串（APIView 的 success() 用原生 json.dumps，datetime 不可序列化）。
    存储为 UTC 的 aware datetime 统一转成北京时间显示，与教师输入一致。"""
    if not dt:
        return None
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone(CN_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _calc_user_exam_score(exam, user_id, paper_index=None):
    """
    动态计算某学生在一场考试中的成绩
    返回 {"total_score", "rank_time"(ACM 罚时秒), "problems": {problem_id: {...}}, "submission_count"}
    """
    try:
        p = ExamParticipation.objects.get(exam=exam, user_id=user_id)
        paper_index = p.paper_index if paper_index is None else paper_index
    except ExamParticipation.DoesNotExist:
        paper_index = 1 if paper_index is None else paper_index
    eps = _exam_problem_queryset(exam, paper_index).select_related("problem")
    subs = Submission.objects.filter(exam=exam, user_id=user_id)
    result = {"total_score": 0, "rank_time": 0, "problems": {}, "submission_count": subs.count(),
              "paper_index": paper_index, "has_participation": False}
    if ExamParticipation.objects.filter(exam=exam, user_id=user_id).exists():
        result["has_participation"] = True
    for ep in eps:
        p_subs = subs.filter(problem_id=ep.problem_id).order_by("create_time")
        info = {"score": 0, "status": None, "ac": False, "submission_count": p_subs.count(),
                "is_late": False, "first_ac_time": None, "error_count": 0}
        if p_subs.exists():
            first_ac = None
            err_before_ac = 0
            best = 0
            for s in p_subs:
                is_late = s.create_time > exam.end_time
                stat = s.statistic_info or {}
                s_score = stat.get("score", 0) or 0
                if exam.rule_type == "ACM":
                    if first_ac is None:
                        if s.result == 0:  # ACCEPTED
                            first_ac = s
                            info["is_late"] = is_late
                            break
                        else:
                            err_before_ac += 1
                else:
                    if s_score > best:
                        best = s_score
                        info["is_late"] = is_late
                    if s.result == 0:
                        info["ac"] = True
            if exam.rule_type == "ACM":
                if first_ac is not None:
                    info["ac"] = True
                    info["status"] = 0
                    info["score"] = ep.score
                    info["first_ac_time"] = _fmt_dt(first_ac.create_time)
                    # 罚时：AC 前错误次数 * 20 分钟
                    penalty = err_before_ac * 20 * 60
                    offset = (first_ac.create_time - exam.start_time).total_seconds()
                    info["rank_offset"] = int(offset + penalty)
                    result["rank_time"] += int(offset + penalty)
                else:
                    info["status"] = p_subs.first().result
                    info["error_count"] = err_before_ac
            else:
                # OI / 作业：按比例折算分值（判题 score 为百分制）
                ratio = max(0, min(100, best)) / 100.0
                info["score"] = int(round(ep.score * ratio))
                info["status"] = 0 if info["ac"] else (p_subs.order_by("-create_time").first().result)
            result["total_score"] += info["score"]
        result["problems"][ep.problem_id] = info
    return result


# ==================== 班级 ====================

class ClassAPI(APIView):
    @login_required
    def get(self, request):
        if _is_teacher(request.user):
            # 教师看板：自己创建的班级
            classes = Class.objects.filter(created_by=request.user, status="active").annotate(
                member_count=Count("members", filter=Q(members__status="active"))).order_by("-create_time")
            data = []
            for c in classes:
                data.append({"id": c.id, "name": c.name, "description": c.description,
                             "semester": c.semester, "join_method": c.join_method,
                             "invite_code": c.invite_code, "create_time": _fmt_dt(c.create_time),
                             "member_count": c.member_count})
            return self.success(data)
        else:
            # 学生：加入的班级
            members = ClassMember.objects.filter(user=request.user, status="active").select_related(
                "class_field", "class_field__created_by")
            data = []
            for m in members:
                c = m.class_field
                data.append({"id": c.id, "name": c.name, "description": c.description,
                             "semester": c.semester, "teacher": c.created_by.username,
                             "create_time": _fmt_dt(c.create_time)})
            return self.success(data)

    @validate_serializer(CreateClassSerializer)
    @login_required
    def post(self, request):
        if not _is_teacher(request.user):
            return self.error("只有教师可以创建班级")
        data = request.data
        cls = Class.objects.create(name=data["name"], description=data.get("description") or "",
                                    semester=data.get("semester") or "",
                                    join_method=data.get("join_method", "code"),
                                    created_by=request.user)
        # 教师自动成为班级成员
        ClassMember.objects.create(class_field=cls, user=request.user, role="assistant", status="active",
                                   real_name=request.user.userprofile.real_name if hasattr(request.user, "userprofile") else None)
        return self.success({"id": cls.id, "invite_code": cls.invite_code})


class ClassDetailAPI(APIView):
    @login_required
    def get(self, request):
        cls = _get_class(request.GET.get("id"))
        if not cls:
            return self.error("班级不存在")
        if not (_is_class_manager(request.user, cls) or _is_class_member(request.user, cls)):
            return self.error("无权访问该班级")
        member_count = ClassMember.objects.filter(class_field=cls, status="active").count()
        data = {"id": cls.id, "name": cls.name, "description": cls.description,
                "semester": cls.semester, "join_method": cls.join_method,
                "create_time": _fmt_dt(cls.create_time), "teacher": cls.created_by.username,
                "member_count": member_count,
                "is_manager": _is_class_manager(request.user, cls),
                "invite_code": cls.invite_code if _is_class_manager(request.user, cls) else None}
        return self.success(data)

    @validate_serializer(EditClassSerializer)
    @login_required
    def put(self, request):
        cls = _get_class(request.data["id"])
        if not cls:
            return self.error("班级不存在")
        if not _is_class_manager(request.user, cls):
            return self.error("无权管理该班级")
        data = request.data
        for field in ("name", "description", "semester", "join_method"):
            if field in data:
                setattr(cls, field, data[field])
        if data.get("reset_invite_code"):
            cls.invite_code = Class.make_invite_code()
        cls.save()
        return self.success({"invite_code": cls.invite_code})

    @login_required
    def delete(self, request):
        cls = _get_class(request.GET.get("id"))
        if not cls:
            return self.error("班级不存在")
        if cls.created_by_id != request.user.id and not request.user.is_admin_role():
            return self.error("无权删除该班级")
        cls.status = "archived"
        cls.save()
        return self.success()


class ClassMemberAPI(APIView):
    @login_required
    def get(self, request):
        cls = _get_class(request.GET.get("class_id"))
        if not cls:
            return self.error("班级不存在")
        if not _is_class_assistant(request.user, cls):
            return self.error("无权查看成员列表")
        status = request.GET.get("status")
        members = ClassMember.objects.filter(class_field=cls).select_related("user").order_by("join_time")
        if status:
            members = members.filter(status=status)
        data = []
        for m in members:
            data.append({"id": m.id, "user_id": m.user_id, "username": m.user.username,
                         "role": m.role, "status": m.status, "student_id": m.student_id,
                         "real_name": m.real_name, "join_time": _fmt_dt(m.join_time)})
        return self.success(data)

    @validate_serializer(JoinClassSerializer)
    @login_required
    def post(self, request):
        """学生通过邀请码加入 / 申请加入（待审批）"""
        data = request.data
        if data.get("invite_code"):
            cls = Class.objects.filter(invite_code=data["invite_code"], status="active").first()
            if not cls:
                return self.error("邀请码无效")
        elif data.get("class_id"):
            cls = _get_class(data["class_id"])
            if not cls:
                return self.error("班级不存在")
        else:
            return self.error("请提供邀请码或班级 ID")
        if _is_class_manager(request.user, cls):
            return self.error("教师无需加入自己创建的班级")
        if ClassMember.objects.filter(class_field=cls, user=request.user).exists():
            return self.error("已申请/加入过该班级")
        if cls.join_method == "code" and data.get("invite_code"):
            member = ClassMember.objects.create(
                class_field=cls, user=request.user, status="active",
                student_id=data.get("student_id") or None,
                real_name=data.get("real_name") or None)
        else:
            member = ClassMember.objects.create(
                class_field=cls, user=request.user, status="pending",
                student_id=data.get("student_id") or None,
                real_name=data.get("real_name") or None)
        return self.success({"id": member.id, "status": member.status})

    @validate_serializer(MemberActionSerializer)
    @login_required
    def put(self, request):
        """教师/助教：审批、设助教、改信息"""
        data = request.data
        cls = _get_class(data["class_id"])
        if not cls:
            return self.error("班级不存在")
        if not _is_class_assistant(request.user, cls):
            return self.error("无权管理成员")
        member = ClassMember.objects.filter(id=data["member_id"], class_field=cls).first()
        if not member:
            return self.error("成员不存在")
        action = data["action"]
        if action == "approve":
            member.status = "active"
        elif action == "reject":
            member.status = "rejected"
        elif action == "remove":
            if member.user_id == cls.created_by_id:
                return self.error("不能移除班级创建者")
            member.delete()
            return self.success()
        elif action == "set_assistant":
            member.role = "assistant"
        elif action == "set_student":
            member.role = "student"
        elif action == "update_info":
            if data.get("student_id"):
                member.student_id = data["student_id"]
            if data.get("real_name"):
                member.real_name = data["real_name"]
        member.save()
        return self.success()

    @login_required
    def delete(self, request):
        """学生退出班级（助教/创建者不可退）"""
        cls = _get_class(request.GET.get("class_id"))
        if not cls:
            return self.error("班级不存在")
        member = ClassMember.objects.filter(class_field=cls, user=request.user).first()
        if not member:
            return self.error("你不在该班级中")
        if member.user_id == cls.created_by_id:
            return self.error("班级创建者不能退出")
        if member.role == "assistant" and not _is_class_manager(request.user, cls):
            return self.error("助教退出请联系教师移除")
        member.delete()
        return self.success()


class ClassDashboardAPI(APIView):
    @login_required
    def get(self, request):
        cls = _get_class(request.GET.get("class_id"))
        if not cls:
            return self.error("班级不存在")
        if not _is_class_manager(request.user, cls):
            return self.error("无权查看班级看板")
        members = ClassMember.objects.filter(class_field=cls, status="active", role="student")
        member_stats = []
        for m in members:
            try:
                profile = UserProfile.objects.get(user=m.user)
                member_stats.append({
                    "member_id": m.id, "username": m.user.username,
                    "student_id": m.student_id, "real_name": m.real_name,
                    "accepted_number": profile.accepted_number,
                    "submission_number": profile.submission_number,
                    "total_score": profile.total_score})
            except UserProfile.DoesNotExist:
                continue
        member_stats.sort(key=lambda x: -(x["accepted_number"] or 0))
        # 班级考试
        exams = Exam.objects.filter(exam_classes__class_field=cls).order_by("-start_time")
        exam_data = []
        for e in exams:
            scores = []
            part_count = ExamParticipation.objects.filter(exam=e).count()
            if e.status == Exam.EXAM_ENDED:
                for m in members:
                    s = _calc_user_exam_score(e, m.user_id)
                    scores.append(s["total_score"])
            avg = round(sum(scores) / len(scores), 1) if scores else None
            exam_data.append({"id": e.id, "title": e.title, "exam_type": e.exam_type,
                              "start_time": _fmt_dt(e.start_time), "end_time": _fmt_dt(e.end_time),
                              "status": e.status, "participant_count": part_count,
                              "avg_score": avg})
        data = {
            "member_count": members.count(),
            "exam_count": exams.count(),
            "avg_accepted": round(sum(x["accepted_number"] or 0 for x in member_stats) / len(member_stats), 1) if member_stats else 0,
            "avg_submission": round(sum(x["submission_number"] or 0 for x in member_stats) / len(member_stats), 1) if member_stats else 0,
            "members": member_stats,
            "exams": exam_data,
        }
        return self.success(data)


# ==================== 考试 ====================

class ExamAPI(APIView):
    @login_required
    def get(self, request):
        if _is_teacher(request.user):
            exam_ids = set(ExamClass.objects.filter(
                class_field__created_by=request.user).values_list("exam_id", flat=True))
            exams = Exam.objects.filter(id__in=exam_ids) if exam_ids else Exam.objects.none()
            if request.GET.get("class_id"):
                cid = int(request.GET.get("class_id"))
                exams = exams.filter(exam_classes__class_field_id=cid)
            exams = exams.order_by("-start_time")
            data = []
            for e in exams:
                data.append({"id": e.id, "title": e.title, "exam_type": e.exam_type,
                             "rule_type": e.rule_type, "start_time": _fmt_dt(e.start_time),
                             "end_time": _fmt_dt(e.end_time), "status": e.status, "visible": e.visible,
                             "problem_count": e.exam_problems.count(),
                             "class_ids": list(e.exam_classes.values_list("class_field_id", flat=True))})
            return self.success(data)
        else:
            # 学生：所在班级的考试
            class_ids = _class_ids_of_user(request.user)
            exams = Exam.objects.filter(exam_classes__class_field_id__in=class_ids, visible=True).distinct().order_by("-start_time")
            data = []
            for e in exams:
                p = ExamParticipation.objects.filter(exam=e, user=request.user).first()
                data.append({"id": e.id, "title": e.title, "exam_type": e.exam_type,
                             "rule_type": e.rule_type, "start_time": _fmt_dt(e.start_time),
                             "end_time": _fmt_dt(e.end_time), "status": e.status,
                             "duration_minutes": e.duration_minutes,
                             "score_public": e.score_public(),
                             "paper_count": e.paper_count,
                             "anti_switch_screen": e.anti_switch_screen,
                             "entered": bool(p),
                             "my_score": None})
                if e.status == Exam.EXAM_ENDED and e.score_public():
                    s = _calc_user_exam_score(e, request.user.id)
                    data[-1]["my_score"] = s["total_score"]
            return self.success(data)

    @validate_serializer(CreateExamSerializer)
    @login_required
    def post(self, request):
        if not _is_teacher(request.user):
            return self.error("只有教师可以创建考试")
        data = request.data
        if data["end_time"] <= data["start_time"]:
            return self.error("结束时间必须晚于开始时间")
        # 校验班级归属
        for cid in data["class_ids"]:
            cls = _get_class(cid)
            if not cls or cls.created_by_id != request.user.id and not request.user.is_admin_role():
                return self.error(f"班级 {cid} 不存在或无权使用")
        exam = Exam.objects.create(
            title=data["title"], description=data.get("description") or "",
            rule_type=data.get("rule_type", "ACM"), exam_type=data.get("exam_type", "exam"),
            start_time=data["start_time"], end_time=data["end_time"],
            late_deadline=data.get("late_deadline"),
            duration_minutes=data.get("duration_minutes", 0),
            max_submissions=data.get("max_submissions", 0),
            score_public_time=data.get("score_public_time"),
            real_time_rank=data.get("real_time_rank", False),
            anti_switch_screen=data.get("anti_switch_screen", False),
            paper_count=1, created_by=request.user)
        for cid in data["class_ids"]:
            ExamClass.objects.create(exam=exam, class_field_id=cid)
        # 组卷
        random_rule = data.get("random_rule") or {}
        if random_rule and random_rule.get("count"):
            # 随机组卷：每套卷独立抽取
            paper_count = min(int(random_rule.get("paper_count", 1) or 1), 10)
            exam.paper_count = paper_count
            exam.random_rule = random_rule
            count = int(random_rule["count"])
            base = Problem.objects.filter(visible=True, contest_id__isnull=True)
            if random_rule.get("difficulty"):
                base = base.filter(difficulty=random_rule["difficulty"])
            if random_rule.get("tags"):
                base = base.filter(tags__name__in=random_rule["tags"]).distinct()
            pids = list(base.values_list("id", flat=True))
            if len(pids) < count:
                exam.delete()
                return self.error(f"满足条件的题目不足（需要 {count} 道，仅找到 {len(pids)} 道）")
            for paper in range(1, paper_count + 1):
                chosen = random.sample(pids, count)
                for i, pid in enumerate(chosen):
                    ExamProblem.objects.create(exam=exam, problem_id=pid, score=100, order=i + 1, paper_index=paper)
            exam.save()
        else:
            for i, item in enumerate(data.get("problems", [])):
                try:
                    problem = Problem.objects.get(id=item["problem_id"], visible=True)
                except Problem.DoesNotExist:
                    exam.delete()
                    return self.error(f"题目 {item['problem_id']} 不存在或不可见")
                ExamProblem.objects.create(exam=exam, problem=problem,
                                           score=item.get("score", 100),
                                           order=i + 1,
                                           paper_index=item.get("paper_index", 0))
        return self.success({"id": exam.id})

    @validate_serializer(EditExamSerializer)
    @login_required
    def put(self, request):
        data = request.data
        try:
            exam = Exam.objects.get(id=data["id"])
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if exam.created_by_id != request.user.id and not request.user.is_admin_role():
            return self.error("无权编辑该考试")
        for field in ("title", "description", "start_time", "end_time", "late_deadline",
                      "duration_minutes", "max_submissions", "score_public_time",
                      "real_time_rank", "anti_switch_screen", "visible"):
            if field in data:
                setattr(exam, field, data[field])
        exam.save()
        if "class_ids" in data:
            ExamClass.objects.filter(exam=exam).delete()
            for cid in data["class_ids"]:
                ExamClass.objects.create(exam=exam, class_field_id=cid)
        if "problems" in data:
            ExamProblem.objects.filter(exam=exam).delete()
            for i, item in enumerate(data["problems"]):
                try:
                    problem = Problem.objects.get(id=item["problem_id"], visible=True)
                except Problem.DoesNotExist:
                    return self.error(f"题目 {item['problem_id']} 不存在")
                ExamProblem.objects.create(exam=exam, problem=problem,
                                           score=item.get("score", 100), order=i + 1,
                                           paper_index=item.get("paper_index", 0))
        return self.success()

    @login_required
    def delete(self, request):
        try:
            exam = Exam.objects.get(id=request.GET.get("id"))
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if exam.created_by_id != request.user.id and not request.user.is_admin_role():
            return self.error("无权删除该考试")
        # 已有学生参与的考试不允许删除，仅能隐藏
        if ExamParticipation.objects.filter(exam=exam).exists():
            exam.visible = False
            exam.save()
            return self.error("已有学生参与，已改为隐藏")
        exam.delete()
        return self.success()


class ExamEnterAPI(APIView):
    """学生进入考试：创建参与记录（抽卷）并返回考试信息"""

    @login_required
    def get(self, request):
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"), visible=True)
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        class_ids = _class_ids_of_user(request.user)
        if not ExamClass.objects.filter(exam=exam, class_field_id__in=class_ids).exists():
            return self.error("你不属于该考试的任何班级")
        if exam.status == Exam.EXAM_NOT_START:
            return self.error("考试尚未开始")
        if exam.status == Exam.EXAM_ENDED:
            can_submit, _ = exam.can_submit_now()
            if not can_submit:
                return self.error("考试已结束")
        # 参与记录（首次进入抽卷 + 计时起点）
        p = ExamParticipation.objects.filter(exam=exam, user=request.user).first()
        if not p:
            if exam.status == Exam.EXAM_NOT_START:
                return self.error("考试尚未开始")
            p = ExamParticipation.objects.create(
                exam=exam, user=request.user,
                paper_index=random.randint(1, exam.paper_count))
        # 限时
        remain_seconds = None
        if exam.duration_minutes > 0:
            deadline = p.first_enter_time + datetime.timedelta(minutes=exam.duration_minutes)
            remain_seconds = max(0, int((deadline - now()).total_seconds()))
            if remain_seconds <= 0 and exam.status == Exam.EXAM_UNDERWAY:
                return self.error("考试时间已用完")
        # 题目列表
        eps = _exam_problem_queryset(exam, p.paper_index).select_related("problem").order_by("order")
        subs = Submission.objects.filter(exam=exam, user_id=request.user.id)
        problems = []
        for ep in eps:
            mine = subs.filter(problem_id=ep.problem_id)
            best = None
            for s in mine:
                stat = s.statistic_info or {}
                s_score = stat.get("score", 0) or 0
                if best is None or s_score > best:
                    best = s_score
            problems.append({"problem_id": ep.problem_id, "_id": ep.problem._id,
                             "title": ep.problem.title, "score": ep.score, "order": ep.order,
                             "my_submission_count": mine.count(),
                             "my_best_score": best, "ac": mine.filter(result=0).exists()})
        data = {"id": exam.id, "title": exam.title, "exam_type": exam.exam_type,
                "rule_type": exam.rule_type, "start_time": _fmt_dt(exam.start_time),
                "end_time": _fmt_dt(exam.end_time), "late_deadline": _fmt_dt(exam.late_deadline),
                "status": exam.status, "paper_index": p.paper_index,
                "remain_seconds": remain_seconds,
                "duration_minutes": exam.duration_minutes,
                "max_submissions": exam.max_submissions,
                "anti_switch_screen": exam.anti_switch_screen,
                "real_time_rank": exam.real_time_rank,
                "my_submissions_used": subs.count(),
                "problems": problems}
        return self.success(data)


class ExamProblemAPI(APIView):
    """考试题目详情（学生做题用）"""

    @login_required
    def get(self, request):
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"), visible=True)
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if not ExamParticipation.objects.filter(exam=exam, user=request.user).exists():
            return self.error("尚未进入考试")
        if exam.status == Exam.EXAM_NOT_START:
            return self.error("考试尚未开始")
        can_submit, _ = exam.can_submit_now()
        if not can_submit:
            return self.error("考试已结束")
        try:
            problem = Problem.objects.get(id=request.GET.get("problem_id"))
        except Problem.DoesNotExist:
            return self.error("题目不存在")
        p = ExamParticipation.objects.get(exam=exam, user=request.user)
        ep = _exam_problem_queryset(exam, p.paper_index).filter(problem=problem).first()
        if not ep:
            return self.error("该题不在你的试卷中")
        data = ProblemSerializer(problem).data
        data["exam_score"] = ep.score
        return self.success(data)


class ExamSubmissionAPI(APIView):
    @validate_serializer(ExamSubmitSerializer)
    @login_required
    def post(self, request):
        data = request.data
        try:
            exam = Exam.objects.get(id=data["exam_id"], visible=True)
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        # 必须已进入考试
        p = ExamParticipation.objects.filter(exam=exam, user=request.user).first()
        if not p:
            return self.error("尚未进入考试")
        # 时间窗
        can_submit, is_late = exam.can_submit_now()
        if not can_submit:
            return self.error("当前不在考试时间内")
        # 限时
        if exam.duration_minutes > 0:
            deadline = p.first_enter_time + datetime.timedelta(minutes=exam.duration_minutes)
            if now() > deadline:
                return self.error("考试时间已用完")
        # 提交次数
        used = Submission.objects.filter(exam=exam, user_id=request.user.id).count()
        if exam.max_submissions and used >= exam.max_submissions:
            return self.error(f"已达到最大提交次数 {exam.max_submissions}")
        # IP 白名单
        if exam.allowed_ip_ranges:
            try:
                user_ip = ipaddress.ip_address(request.session.get("ip"))
                if not any(user_ip in ipaddress.ip_network(cidr, strict=False)
                           for cidr in exam.allowed_ip_ranges):
                    return self.error("当前 IP 不在考试允许范围内")
            except ValueError:
                return self.error("IP 校验失败")
        # 题目必须在试卷内
        ep = _exam_problem_queryset(exam, p.paper_index).filter(problem_id=data["problem_id"]).first()
        if not ep:
            return self.error("该题不在你的试卷中")
        problem = ep.problem
        if data["language"] not in problem.languages:
            return self.error(f"{data['language']} 不是该题允许的语言")
        submission = Submission.objects.create(
            user_id=request.user.id, username=request.user.username,
            language=data["language"], code=data["code"],
            problem_id=problem.id, ip=request.session.get("ip"), exam=exam)
        judge_task.send(submission.id, problem.id)
        return self.success({"submission_id": submission.id, "is_late": is_late,
                             "submissions_used": used + 1})

    @login_required
    def get(self, request):
        """学生在某场考试中的提交列表"""
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"))
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if not ExamParticipation.objects.filter(exam=exam, user=request.user).exists():
            return self.error("尚未进入考试")
        subs = Submission.objects.filter(exam=exam, user_id=request.user.id).select_related("problem").order_by("-create_time")
        problem_id = request.GET.get("problem_id")
        if problem_id:
            subs = subs.filter(problem_id=problem_id)
        data = [{"id": s.id, "problem_id": s.problem_id, "problem__id": s.problem._id,
                 "result": s.result, "language": s.language, "create_time": _fmt_dt(s.create_time),
                 "statistic_info": s.statistic_info} for s in subs[:200]]
        return self.success(data)


class ExamScoreAPI(APIView):
    @login_required
    def get(self, request):
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"))
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        is_teacher_view = exam.created_by_id == request.user.id or request.user.is_admin_role()
        if not is_teacher_view:
            # 助教可看
            if ExamClass.objects.filter(exam=exam, class_field__members__user=request.user,
                                        class_field__members__role="assistant",
                                        class_field__members__status="active").exists():
                is_teacher_view = True
        eps = exam.exam_problems.all().select_related("problem").order_by("order")
        problems_meta = [{"problem_id": ep.problem_id, "_id": ep.problem._id,
                          "title": ep.problem.title, "score": ep.score} for ep in eps]
        if is_teacher_view:
            rows = []
            members = ClassMember.objects.filter(
                class_field__in=Class.objects.filter(class_exams__exam=exam), status="active", role="student").select_related("user").distinct()
            for m in members:
                s = _calc_user_exam_score(exam, m.user_id)
                rows.append({"username": m.user.username, "student_id": m.student_id,
                             "real_name": m.real_name, "total_score": s["total_score"],
                             "rank_time": s["rank_time"], "submission_count": s["submission_count"],
                             "has_participation": s["has_participation"],
                             "problems": s["problems"]})
            # 排名（ACM 按总分降序+罚时升序；OI 按总分降序）
            rows.sort(key=lambda r: (-r["total_score"], r["rank_time"]))
            for i, r in enumerate(rows):
                r["rank"] = i + 1
            return self.success({"is_teacher": True, "score_public": exam.score_public(),
                                "problems": problems_meta, "rows": rows})
        else:
            # 学生：公布前仅返回总体状态
            if not exam.score_public():
                return self.success({"is_teacher": False, "score_public": False,
                                     "problems": [], "rows": []})
            s = _calc_user_exam_score(exam, request.user.id)
            # 实时排名（需 real_time_rank 且进行中）
            rank = None
            if exam.real_time_rank or exam.status == Exam.EXAM_ENDED:
                class_ids = _class_ids_of_user(request.user)
                members = ClassMember.objects.filter(
                    class_field_id__in=class_ids, status="active", role="student").values_list("user_id", flat=True)
                scores = [_calc_user_exam_score(exam, uid)["total_score"] for uid in members]
                rank = 1 + sum(1 for x in scores if x > s["total_score"])
            return self.success({"is_teacher": False, "score_public": True,
                                 "problems": problems_meta, "rank": rank,
                                 "total_score": s["total_score"],
                                 "problems_detail": s["problems"],
                                 "submission_count": s["submission_count"]})


class ExamScoreExportAPI(APIView):
    """成绩单导出 CSV（Excel 可直接打开）"""

    @login_required
    def get(self, request):
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"))
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if exam.created_by_id != request.user.id and not request.user.is_admin_role():
            return self.error("无权导出成绩")
        eps = exam.exam_problems.all().select_related("problem").order_by("order")
        members = ClassMember.objects.filter(
            class_field__in=Class.objects.filter(class_exams__exam=exam), status="active", role="student").select_related("user").distinct()
        rows = []
        for m in members:
            s = _calc_user_exam_score(exam, m.user_id)
            rows.append((m, s))
        rows.sort(key=lambda x: -x[1]["total_score"])
        header = ["排名", "用户名", "学号", "姓名", "总分"] + \
                 [f"{ep.problem._id}({ep.score}分)" for ep in eps] + ["提交次数"]
        lines = [",".join(header)]
        for rank, (m, s) in enumerate(rows, 1):
            cells = [str(rank), m.user.username,
                     (m.student_id or "").replace(",", " "),
                     (m.real_name or "").replace(",", " "),
                     str(s["total_score"])]
            for ep in eps:
                info = s["problems"].get(ep.problem_id, {})
                cells.append(str(info.get("score", 0)))
            cells.append(str(s["submission_count"]))
            lines.append(",".join(cells))
        content = "\ufeff" + "\n".join(lines)  # UTF-8 BOM 便于 Excel 识别中文
        resp = HttpResponse(content, content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = f'attachment; filename="exam_{exam.id}_scores.csv"'
        return resp


class ExamMonitorAPI(APIView):
    @validate_serializer(MonitorReportSerializer)
    @login_required
    def post(self, request):
        """学生端上报切屏等监考事件"""
        data = request.data
        try:
            exam = Exam.objects.get(id=data["exam_id"], visible=True)
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if not ExamParticipation.objects.filter(exam=exam, user=request.user).exists():
            return self.error("尚未进入考试")
        if exam.status != Exam.EXAM_UNDERWAY:
            return self.error("考试未在进行中")
        ExamMonitorLog.objects.create(exam=exam, user=request.user,
                                      event_type=data["event_type"], detail=data.get("detail") or "")
        return self.success()

    @login_required
    def get(self, request):
        """教师查看监考统计"""
        try:
            exam = Exam.objects.get(id=request.GET.get("exam_id"))
        except Exam.DoesNotExist:
            return self.error("考试不存在")
        if exam.created_by_id != request.user.id and not request.user.is_admin_role():
            return self.error("无权查看监考数据")
        logs = ExamMonitorLog.objects.filter(exam=exam).select_related("user")
        data = []
        for log in logs[:500]:
            data.append({"username": log.user.username, "event_type": log.event_type,
                         "detail": log.detail, "create_time": _fmt_dt(log.create_time)})
        return self.success(data)
