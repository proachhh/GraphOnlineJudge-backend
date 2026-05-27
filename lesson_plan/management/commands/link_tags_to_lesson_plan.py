from django.core.management.base import BaseCommand
from lesson_plan.models import LessonPlan, LessonPlanProblem
from problem.models import Problem, ProblemTag


class Command(BaseCommand):
    help = '自动将标签匹配的题目关联到教案'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true', default=False,
            help='只预览不实际写入'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        total_linked = 0

        lesson_plans = LessonPlan.objects.filter(visible=True)
        self.stdout.write(f'共 {lesson_plans.count()} 份教案')

        for lp in lesson_plans:
            title = lp.title.strip()
            if not title:
                continue

            tag_names = self._find_matching_tags(title)
            if not tag_names:
                self.stdout.write(f'  ⚠ {title}: 未找到匹配的知识点标签')
                continue

            existing_problem_ids = set(
                LessonPlanProblem.objects.filter(lesson_plan=lp)
                .values_list('problem_id', flat=True)
            )

            matched_problems = Problem.objects.filter(
                visible=True, tags__name__in=tag_names
            ).distinct()

            new_count = 0
            current_order = LessonPlanProblem.objects.filter(
                lesson_plan=lp
            ).count()

            for problem in matched_problems:
                if problem.id in existing_problem_ids:
                    continue

                if not dry_run:
                    LessonPlanProblem.objects.create(
                        lesson_plan=lp,
                        problem=problem,
                        order=current_order + new_count
                    )
                new_count += 1

            if new_count > 0:
                total_linked += new_count
                tag_str = ', '.join(tag_names)
                self.stdout.write(
                    f'  ✓ {title}: 匹配标签 [{tag_str}] → '
                    f'关联了 {new_count} 道题目'
                )
            else:
                self.stdout.write(
                    f'  - {title}: 标签匹配但无新增题目'
                )

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f'\n[预览模式] 将关联 {total_linked} 条记录'
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f'\n完成！共为教案关联了 {total_linked} 道题目'
            ))

    def _find_matching_tags(self, title):
        all_tags = ProblemTag.objects.values_list('name', flat=True)
        matched = []

        title_clean = title.replace('#', '').replace(' ', '').strip()

        for tag in all_tags:
            tag_clean = tag.replace(' ', '').strip()
            if not tag_clean:
                continue

            if tag_clean == title_clean:
                matched.append(tag)
                continue

            if tag_clean in title_clean or title_clean in tag_clean:
                matched.append(tag)
                continue

        return matched
