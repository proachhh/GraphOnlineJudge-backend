from django.db import models

from account.models import User
from problem.models import Problem
from utils.models import JSONField, RichTextField


class ExerciseSet(models.Model):
    """试题集"""
    title = models.TextField()
    description = models.TextField(null=True)
    # knowledge topic name, e.g. "动态规划"
    topic = models.TextField()
    # Low / Mid / High
    difficulty = models.TextField(default="Mid")
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)
    is_published = models.BooleanField(default=False)
    # minutes, null = no limit
    time_limit = models.IntegerField(null=True)
    # max attempts per user, null = unlimited
    max_attempts = models.IntegerField(null=True)
    # whether students can see their score after submission
    show_score = models.BooleanField(default=True)
    # whether students can see the ranking
    show_ranking = models.BooleanField(default=True)
    # store extra info like tags
    meta = JSONField(default=dict)

    class Meta:
        db_table = "exercise_set"
        ordering = ("-create_time",)


class ExerciseQuestion(models.Model):
    """试题集中的题目"""
    exercise_set = models.ForeignKey(ExerciseSet, related_name='questions', on_delete=models.CASCADE)
    # linked OJ problem, null for pure quiz questions
    problem = models.ForeignKey(Problem, null=True, on_delete=models.SET_NULL)
    order = models.IntegerField(default=0)
    # choice / code / short_answer
    question_type = models.TextField(default='choice')
    content = RichTextField()
    # for choice questions: [{"key": "A", "text": "..."}]
    choices = JSONField(default=list)
    # for choice: "A"; for code: expected output; null for AI-graded
    correct_answer = models.TextField(null=True)
    score = models.IntegerField(default=10)
    explanation = models.TextField(null=True)

    class Meta:
        db_table = "exercise_question"
        ordering = ("order", "id")


class ExerciseSubmission(models.Model):
    """作答记录"""
    exercise_set = models.ForeignKey(ExerciseSet, related_name='submissions', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='exercise_submissions', on_delete=models.CASCADE)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)
    total_score = models.IntegerField(default=0)
    # in_progress / submitted / graded
    status = models.TextField(default='in_progress')

    class Meta:
        db_table = "exercise_submission"
        ordering = ("-create_time",)


class ExerciseAnswer(models.Model):
    """每题作答"""
    submission = models.ForeignKey(ExerciseSubmission, related_name='answers', on_delete=models.CASCADE)
    question = models.ForeignKey(ExerciseQuestion, on_delete=models.CASCADE)
    answer = models.TextField(null=True)
    is_correct = models.BooleanField(null=True)
    score = models.IntegerField(default=0)
    ai_feedback = models.TextField(null=True)
    graded_at = models.DateTimeField(null=True)

    class Meta:
        db_table = "exercise_answer"
        unique_together = (("submission", "question"),)


class BossExam(models.Model):
    """Boss动态试卷"""
    title = models.TextField()
    description = models.TextField(null=True)
    # list of topic names defining the knowledge area
    topic_area = JSONField(default=list)
    # the boss / knowledge convergence node
    boss_topic = models.TextField()
    difficulty = models.TextField(default="Mid")
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    create_time = models.DateTimeField(auto_now_add=True)
    is_published = models.BooleanField(default=False)
    # minutes
    time_limit = models.IntegerField(null=True)
    passing_score = models.IntegerField(default=60)

    class Meta:
        db_table = "boss_exam"
        ordering = ("-create_time",)


class BossQuestion(models.Model):
    """Boss试卷题目"""
    exam = models.ForeignKey(BossExam, related_name='questions', on_delete=models.CASCADE)
    problem = models.ForeignKey(Problem, null=True, on_delete=models.SET_NULL)
    order = models.IntegerField(default=0)
    question_type = models.TextField(default='choice')
    content = RichTextField()
    choices = JSONField(default=list)
    correct_answer = models.TextField(null=True)
    score = models.IntegerField(default=10)
    # which topic in the area this question tests
    topic = models.TextField(null=True)

    class Meta:
        db_table = "boss_question"
        ordering = ("order", "id")


class BossSubmission(models.Model):
    """Boss试卷作答"""
    exam = models.ForeignKey(BossExam, related_name='submissions', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='boss_submissions', on_delete=models.CASCADE)
    create_time = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)
    total_score = models.IntegerField(default=0)
    # in_progress / submitted / graded
    status = models.TextField(default='in_progress')
    ai_evaluation = models.TextField(null=True)
    passed = models.BooleanField(default=False)

    class Meta:
        db_table = "boss_submission"
        ordering = ("-create_time",)


class BossAnswer(models.Model):
    """Boss每题作答"""
    submission = models.ForeignKey(BossSubmission, related_name='answers', on_delete=models.CASCADE)
    question = models.ForeignKey(BossQuestion, on_delete=models.CASCADE)
    answer = models.TextField(null=True)
    is_correct = models.BooleanField(null=True)
    score = models.IntegerField(default=0)
    ai_feedback = models.TextField(null=True)

    class Meta:
        db_table = "boss_answer"
        unique_together = (("submission", "question"),)
