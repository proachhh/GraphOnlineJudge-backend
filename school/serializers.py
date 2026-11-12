import pytz
from rest_framework import serializers

# 教师输入的时间均为北京时间字符串（naive），显式指定解析时区，
# 避免 settings.TIME_ZONE='UTC' 导致存储时间比教师输入晚 8 小时
CN_TZ = pytz.timezone("Asia/Shanghai")
DT_KWARGS = {"default_timezone": CN_TZ}


class CreateClassSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=64)
    description = serializers.CharField(required=False, allow_blank=True, default="")
    semester = serializers.CharField(required=False, allow_blank=True, default="")
    join_method = serializers.ChoiceField(choices=("code", "approval"), default="code")


class EditClassSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField(required=False, max_length=64)
    description = serializers.CharField(required=False, allow_blank=True)
    semester = serializers.CharField(required=False, allow_blank=True)
    join_method = serializers.ChoiceField(required=False, choices=("code", "approval"))
    # 重新生成邀请码
    reset_invite_code = serializers.BooleanField(required=False, default=False)


class JoinClassSerializer(serializers.Serializer):
    invite_code = serializers.CharField(required=False, max_length=16)
    class_id = serializers.IntegerField(required=False)
    # 申请时填写的学号/真实姓名（可选）
    student_id = serializers.CharField(required=False, allow_blank=True, default="")
    real_name = serializers.CharField(required=False, allow_blank=True, default="")


class MemberActionSerializer(serializers.Serializer):
    class_id = serializers.IntegerField()
    member_id = serializers.IntegerField()
    action = serializers.ChoiceField(choices=("approve", "reject", "remove",
                                             "set_assistant", "set_student",
                                             "update_info"))
    student_id = serializers.CharField(required=False, allow_blank=True, default="")
    real_name = serializers.CharField(required=False, allow_blank=True, default="")


class ExamProblemItemSerializer(serializers.Serializer):
    problem_id = serializers.IntegerField()
    score = serializers.IntegerField(default=100, min_value=1)
    paper_index = serializers.IntegerField(default=0, min_value=0)


class CreateExamSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=128)
    description = serializers.CharField(required=False, allow_blank=True, default="")
    rule_type = serializers.ChoiceField(choices=("ACM", "OI"), default="ACM")
    exam_type = serializers.ChoiceField(choices=("exam", "homework"), default="exam")
    start_time = serializers.DateTimeField(**DT_KWARGS)
    end_time = serializers.DateTimeField(**DT_KWARGS)
    late_deadline = serializers.DateTimeField(required=False, allow_null=True, default=None, **DT_KWARGS)
    duration_minutes = serializers.IntegerField(default=0, min_value=0)
    max_submissions = serializers.IntegerField(default=0, min_value=0)
    score_public_time = serializers.DateTimeField(required=False, allow_null=True, default=None, **DT_KWARGS)
    real_time_rank = serializers.BooleanField(default=False)
    anti_switch_screen = serializers.BooleanField(default=False)
    class_ids = serializers.ListField(child=serializers.IntegerField(), allow_empty=False)
    # 手动组卷题目列表
    problems = serializers.ListField(
        child=ExamProblemItemSerializer(), required=False, default=list)
    # 随机组卷规则（非空时按难度/标签抽题）：difficulty Low/Mid/High，tags 列表，count 数量
    random_rule = serializers.DictField(required=False, default=dict)


class EditExamSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField(required=False, max_length=128)
    description = serializers.CharField(required=False, allow_blank=True)
    start_time = serializers.DateTimeField(required=False, **DT_KWARGS)
    end_time = serializers.DateTimeField(required=False, **DT_KWARGS)
    late_deadline = serializers.DateTimeField(required=False, allow_null=True, **DT_KWARGS)
    duration_minutes = serializers.IntegerField(required=False, min_value=0)
    max_submissions = serializers.IntegerField(required=False, min_value=0)
    score_public_time = serializers.DateTimeField(required=False, allow_null=True, **DT_KWARGS)
    real_time_rank = serializers.BooleanField(required=False)
    anti_switch_screen = serializers.BooleanField(required=False)
    class_ids = serializers.ListField(child=serializers.IntegerField(), required=False)
    problems = serializers.ListField(child=ExamProblemItemSerializer(), required=False)
    visible = serializers.BooleanField(required=False)


class ExamSubmitSerializer(serializers.Serializer):
    exam_id = serializers.IntegerField()
    problem_id = serializers.IntegerField()
    code = serializers.CharField()
    language = serializers.CharField()


class MonitorReportSerializer(serializers.Serializer):
    exam_id = serializers.IntegerField()
    event_type = serializers.ChoiceField(choices=("switch_screen",))
    detail = serializers.CharField(required=False, allow_blank=True, default="")
