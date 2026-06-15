"""
重建 Neo4j 中 Topic ← Problem 的 BELONGS_TO 关系。
解决 rebuild_kg 后中文 Topic 名与英文 ProblemTag 名不匹配导致的关联缺失问题。
"""
from django.core.management.base import BaseCommand
from utils.neo4j_client import neo4j_client
from problem.models import Problem, ProblemTag
from agents.path_planning_agent import TOPIC_TO_TAG_KEYWORDS


def get_matching_tag_names(topic_name: str, all_tags: list) -> list:
    if topic_name in all_tags:
        return [topic_name]

    topic_lower = topic_name.strip().lower()
    matches = set()

    for tag in all_tags:
        if tag.strip().lower() == topic_lower:
            matches.add(tag)

    if not matches:
        for tag in all_tags:
            tag_lower = tag.strip().lower()
            if (topic_lower in tag_lower or tag_lower in topic_lower) and \
               (len(topic_lower) >= 3 or len(tag_lower) >= 3):
                matches.add(tag)

    if not matches:
        topic_words = set(topic_lower.replace('-', ' ').replace('_', ' ').split())
        for tag in all_tags:
            tag_words = set(tag.strip().lower().replace('-', ' ').replace('_', ' ').split())
            if topic_words and tag_words:
                overlap = topic_words & tag_words
                if len(overlap) >= len(topic_words) * 0.5 or len(overlap) >= 2:
                    matches.add(tag)

    # 中文关键词映射
    if not matches and topic_name in TOPIC_TO_TAG_KEYWORDS:
        keywords = TOPIC_TO_TAG_KEYWORDS[topic_name]
        for tag in all_tags:
            tag_lower_compact = tag.strip().lower().replace(' ', '').replace('-', '').replace('_', '')
            for kw in keywords:
                kw_compact = kw.lower().replace(' ', '').replace('-', '').replace('_', '')
                if tag_lower_compact == kw_compact:
                    matches.add(tag)
                    break
                if kw_compact in tag_lower_compact and len(kw_compact) >= 3:
                    matches.add(tag)
                    break

    return list(matches)[:5]


class Command(BaseCommand):
    help = '为 Neo4j Topic 节点重建 BELONGS_TO 关系'

    def handle(self, *args, **options):
        client = neo4j_client

        # 获取所有 Topic 节点
        topics = client.run_query("MATCH (t:Topic) RETURN t.name AS name")
        topic_names = [t['name'] for t in topics]
        self.stdout.write(f'Neo4j 中共有 {len(topic_names)} 个 Topic 节点')

        # 获取所有 ProblemTag
        all_tags = list(ProblemTag.objects.values_list('name', flat=True).distinct())
        self.stdout.write(f'PostgreSQL 中共有 {len(all_tags)} 个 ProblemTag')

        # 获取所有 Problem ID
        all_problem_ids = list(
            Problem.objects.filter(visible=True, contest__isnull=True)
            .values_list('id', flat=True)
        )

        created = 0
        skipped = 0
        no_match = 0

        for topic_name in topic_names:
            tag_names = get_matching_tag_names(topic_name, all_tags)

            if not tag_names:
                no_match += 1
                continue

            # 找到该 Topic 关联的所有 Problem
            problem_ids = list(
                Problem.objects
                .filter(tags__name__in=tag_names, visible=True, contest__isnull=True)
                .values_list('id', flat=True)
                .distinct()
            )

            if not problem_ids:
                no_match += 1
                continue

            for pid in problem_ids:
                try:
                    result = client.run_query(
                        """
                        MATCH (p:Problem {problem_id: $pid})
                        MATCH (t:Topic {name: $name})
                        WHERE NOT EXISTS((p)-[:BELONGS_TO]->(t))
                        CREATE (p)-[:BELONGS_TO]->(t)
                        RETURN count(*) AS cnt
                        """,
                        {'pid': pid, 'name': topic_name}
                    )
                    if result and result[0].get('cnt', 0) > 0:
                        created += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1

            if created % 50 == 0 and created > 0:
                self.stdout.write(f'  已创建 {created} 条关系...')

        self.stdout.write('=' * 50)
        self.stdout.write(f'创建 BELONGS_TO 关系: {created} 条')
        self.stdout.write(f'跳过 (已存在或无Problem节点): {skipped} 条')
        self.stdout.write(f'无法匹配 Topic: {no_match} 个')
        self.stdout.write(self.style.SUCCESS('BELONGS_TO 关系重建完成!'))
