from django.db import models
from utils.models import RichTextField
from utils.shortcuts import rand_str


class Feedback(models.Model):
    id = models.CharField(max_length=32, default=rand_str, primary_key=True, db_index=True)
    user = models.ForeignKey("account.User", on_delete=models.CASCADE)
    username = models.CharField(max_length=32)
    title = models.CharField(max_length=128)
    content = RichTextField(null=True, blank=True)
    resolved = models.BooleanField(default=False)
    admin_note = models.TextField(null=True, blank=True)
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "feedback"
        ordering = ("-create_time",)

    def __str__(self):
        return self.title
