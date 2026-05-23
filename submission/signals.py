from django.db.models.signals import post_save
from django.dispatch import receiver
from submission.models import Submission
from submission.tasks import sync_submission_to_neo4j

@receiver(post_save, sender=Submission)
def submission_post_save(sender, instance, created, **kwargs):
    if created:
        sync_submission_to_neo4j.send(str(instance.id))
        from knowledge_graph.tasks import update_user_mastery
        update_user_mastery.send(instance.user_id)
        _try_update_profile(instance)


def _try_update_profile(instance):
    try:
        from problem.models import ProblemTag
        from agents.master_agent import master_agent

        tags = list(ProblemTag.objects.filter(
            problem_id=instance.problem_id
        ).values_list('name', flat=True))

        if not tags:
            return

        event = {
            'result': instance.result,
            'tags': tags,
            'problem_title': instance.problem.title if instance.problem_id else '',
        }
        master_agent.handle_submission_event(instance.user_id, event)
    except Exception:
        pass