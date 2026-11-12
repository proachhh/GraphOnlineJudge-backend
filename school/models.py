from django.db import models
from django.utils.timezone import now
from utils.models import JSONField, RichTextField
from utils.shortcuts import rand_str

from account.models import User
from problem.models import Problem


def make_invite_code():
    return rand_str(8)


class Class(models.Model):
    """班级：教师创建的教学组织单元"""
    name = models.TextField()
    description = RichTextField(null=True)
    semester = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="created_classes")
    # 邀请码 / 教师审批
    JOIN_METHOD = {"code": "code", "approval": "approval"}
    join_method = models.TextField(default="code")
    invite_code = models.TextField(default=make_invite_code, unique=True)
    # active 正常 / archived 归档
    status = models.TextField(default="active")
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "school_class"
        ordering = ("-create_time",)


class ClassMember(models.Model):
    """班级成员：学生 / 助教"""
    class_field = models.ForeignKey(Class, on_delete=models.CASCADE, related_name="members")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ROLE = {"student": "student", "assistant": "assistant"}
    role = models.TextField(default="student")
    # pending(待审批) / active(正常) / rejected(被拒)
    status = models.TextField(default="active")
    # 冗余学号/真实姓名（以 UserProfile 为准，入班时留空则取个人资料）
    student_id = models.TextField(null=True, blank=True)
    real_name = models.TextField(null=True, blank=True)
    join_time = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # 未填写学号/姓名时自动取个人资料
        if not self.student_id and self.user_id:
            profile = self.user.userprofile if hasattr(self.user, "userprofile") else None
            if profile:
                if not self.student_id:
                    self.student_id = profile.student_id
                if not self.real_name:
                    self.real_name = profile.real_name
        super().save(*args, **kwargs)

    class Meta:
        db_table = "school_class_member"
        unique_together = (("class_field", "user"),)


class Exam(models.Model):
    """考试 / 作业（homework 为作业模式：允许迟交、多次提交不排名）"""
    title = models.TextField()
    description = RichTextField(null=True)
    # ACM: 通过计满分+罚时; OI: 按测试点得分
    rule_type = models.TextField(default="ACM")
    exam_type = models.TextField(default="exam")  # exam / homework
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    # 作业模式：迟交截止时间（end_time 之后仍可提交到此时间，标记迟交）
    late_deadline = models.DateTimeField(null=True, blank=True)
    # >0 时每人限时分钟数（从首次进入考试起算）
    duration_minutes = models.IntegerField(default=0)
    # 0 = 不限提交次数
    max_submissions = models.IntegerField(default=0)
    # 成绩公布时间（null = 考试结束后即公布）
    score_public_time = models.DateTimeField(null=True, blank=True)
    # 考试期间是否显示实时排名
    real_time_rank = models.BooleanField(default=False)
    # 监考：切屏检测
    anti_switch_screen = models.BooleanField(default=False)
    # 监考：IP 白名单（CIDR 列表）
    allowed_ip_ranges = JSONField(default=list)
    # 多套卷：paper_count > 1 时学生首次进入随机分配套卷
    paper_count = models.IntegerField(default=1)
    # 随机组卷配置：{"difficulty": "Low", "tags": ["贪心"], "count": 5}
    # 非空时按规则从题库抽题组卷（每套卷独立随机）
    random_rule = JSONField(default=dict)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    visible = models.BooleanField(default=True)
    create_time = models.DateTimeField(auto_now_add=True)

    EXAM_NOT_START = "1"
    EXAM_UNDERWAY = "0"
    EXAM_ENDED = "-1"

    @property
    def status(self):
        if self.start_time > now():
            return self.EXAM_NOT_START
        elif self.end_time < now():
            return self.EXAM_ENDED
        return self.EXAM_UNDERWAY

    def score_public(self):
        """成绩是否已公布"""
        if self.score_public_time is None:
            return self.status == self.EXAM_ENDED
        return now() >= self.score_public_time

    def can_submit_now(self):
        """当前是否允许提交（考试：进行中；作业：进行中或迟交窗口内）"""
        if self.status == self.EXAM_UNDERWAY:
            return True, False
        if self.exam_type == "homework" and self.late_deadline and now() <= self.late_deadline:
            return True, True  # 迟交
        return False, False

    class Meta:
        db_table = "school_exam"
        ordering = ("-start_time",)


class ExamClass(models.Model):
    """考试与班级的多对多关联（支持多班联考）"""
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="exam_classes")
    class_field = models.ForeignKey(Class, on_delete=models.CASCADE, related_name="class_exams")

    class Meta:
        db_table = "school_exam_class"
        unique_together = (("exam", "class_field"),)


class ExamProblem(models.Model):
    """考试题目：引用公开题库的题目，不复制副本"""
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="exam_problems")
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE)
    score = models.IntegerField(default=100)
    order = models.IntegerField(default=1)
    # paper_index=0 表示所有套卷公共；>0 表示仅第 N 套卷包含（多套卷时用于套卷专属题）
    paper_index = models.IntegerField(default=0)

    class Meta:
        db_table = "school_exam_problem"
        ordering = ("order",)


class ExamParticipation(models.Model):
    """学生参加考试的记录：首次进入时间（限时用）、分配的套卷"""
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="participations")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    paper_index = models.IntegerField(default=1)
    first_enter_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "school_exam_participation"
        unique_together = (("exam", "user"),)


class ExamMonitorLog(models.Model):
    """监考日志：切屏等行为上报"""
    id = models.TextField(default=rand_str, primary_key=True, db_index=True)
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="monitor_logs")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    event_type = models.TextField()  # switch_screen
    detail = models.TextField(null=True, blank=True)
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "school_exam_monitor_log"
        ordering = ("-create_time",)
