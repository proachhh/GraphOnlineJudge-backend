from django.contrib import admin

from .models import (BossAnswer, BossExam, BossQuestion, BossSubmission,
                     ExerciseAnswer, ExerciseQuestion, ExerciseSet, ExerciseSubmission)

admin.site.register(ExerciseSet)
admin.site.register(ExerciseQuestion)
admin.site.register(ExerciseSubmission)
admin.site.register(ExerciseAnswer)
admin.site.register(BossExam)
admin.site.register(BossQuestion)
admin.site.register(BossSubmission)
admin.site.register(BossAnswer)
