import os
import re
import json

from django.core.management.base import BaseCommand
from utils.neo4j_client import neo4j_client


FOUNDATIONAL_KEYWORDS = [
    '线性表', '顺序表', '链表', '单链表', '循环链表', '双向链表',
    '栈', '队列', '数组', '树', '图', '二叉树', '排序', '查找',
    '哈希', '字符串', '递归', '循环', '分支',
]

ADVANCED_KEYWORDS = [
    'KMP', '哈夫曼', '平衡', '最小生成', '最短路径', '拓扑排序',
    '优先队列', '线索二叉', '双端队列',
]

DIFFICULTY_MAP = {
    'O(1)': 1,
    'O(log': 2,
    'O(n)': 3,
    'O(n+m)': 3,
    'O(n log n)': 4,
    'O(n^2)': 4,
    'O(n²)': 4,
    'O(n^3)': 5,
    'O(2^n)': 5,
    'O(2ⁿ)': 5,
    'O(n!)': 5,
}


def parse_filenames_to_topics(directory: str) -> list:
    if not os.path.isdir(directory):
        return []

    topics = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.md'):
            continue
        name = fname
        name = re.sub(r'^##\s*', '', name)
        name = re.sub(r'\.md$', '', name)
        name = name.strip()
        if name and not name.startswith('教案'):
            topics.append({
                'name': name,
                'filename': fname,
                'filepath': os.path.join(directory, fname),
            })
    return topics


def read_lesson_plans_from_db() -> list:
    """从数据库 lesson_plan 表读取教案（content 为 Markdown），返回与 parse_filenames_to_topics 同构的列表"""
    from lesson_plan.models import LessonPlan
    topics = []
    for lp in LessonPlan.objects.all():
        name = (lp.title or '').strip()
        name = re.sub(r'^##\s*', '', name)
        name = re.sub(r'\.md$', '', name).strip()
        if not name or name.startswith('教案'):
            continue
        topics.append({
            'name': name,
            'filename': f'{name}.md',
            'filepath': None,
            'content': lp.content or '',
            'source_file': f'db:lesson_plan:{lp.id}',
        })
    return topics


def read_file_content(filepath: str) -> str:
    if not filepath:
        return ''
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''


def parse_complexity_from_content(content: str) -> int:
    found_complexities = []
    for pattern, level in sorted(DIFFICULTY_MAP.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        if pattern in content:
            found_complexities.append(level)

    if found_complexities:
        return max(found_complexities)

    complexity_section = ''
    match = re.search(
        r'###\s*[二2三四五六].*?(?:操作|方法|复杂度|[Oo]\().*?\n(.*?)(?=\n###|\n##|\Z)',
        content, re.DOTALL
    )
    if not match:
        match = re.search(
            r'##\s*[二2三四五六].*?(?:操作|方法|复杂度|[Oo]\().*?\n(.*?)(?=\n##|\Z)',
            content, re.DOTALL
        )
    if match:
        complexity_section = match.group(0)

    for pattern, level in sorted(DIFFICULTY_MAP.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        if pattern in complexity_section:
            found_complexities.append(level)

    if found_complexities:
        return max(found_complexities)

    return 3


def parse_importance(topic_name: str) -> int:
    name = topic_name.replace(' ', '')

    for kw in FOUNDATIONAL_KEYWORDS:
        if kw.replace(' ', '') in name and '应用' not in name:
            return 5

    for kw in ADVANCED_KEYWORDS:
        if kw.replace(' ', '') in name:
            return 3

    if '基本概念' in name or '概念' in name:
        return 4

    if '存储' in name or '遍历' in name:
        return 4

    if '应用' in name:
        return 3

    return 3


def parse_relationships(content: str) -> dict:
    prereqs = []
    successors = []

    rel_section = ''
    match = re.search(
        r'与其他知识点的关系.*?\n(.*?)(?=\n## [^\n]|\Z)',
        content, re.DOTALL
    )
    if match:
        rel_section = match.group(0)

    prereq_match = re.search(
        r'\*{0,2}前置依赖\*{0,2}[：:]\s*(.+?)(?:\n-|\n#|\Z)',
        rel_section
    )
    if prereq_match:
        raw = prereq_match.group(1).strip()
        prereqs = [
            x.strip().rstrip('。，,;')
            for x in re.split(r'[、,，;；]', raw)
            if x.strip() and len(x.strip()) >= 2
        ]

    succ_match = re.search(
        r'\*{0,2}后续拓展\*{0,2}[：:]\s*(.+?)(?:\n-|\n#|\Z)',
        rel_section
    )
    if succ_match:
        raw = succ_match.group(1).strip()
        successors = [
            x.strip().rstrip('。，,;')
            for x in re.split(r'[、,，;；]', raw)
            if x.strip() and len(x.strip()) >= 2
        ]

    return {
        'prerequisites': prereqs,
        'successors': successors,
    }


def fuzzy_match_topic_name(search_name: str, all_topics: list) -> str:
    search_clean = search_name.replace(' ', '').replace('的', '')

    for t in all_topics:
        t_clean = t.replace(' ', '').replace('的', '')
        if search_clean == t_clean:
            return t
        if search_clean in t_clean or t_clean in search_clean:
            return t

    return ''


class Command(BaseCommand):
    help = '从 /teach/ 教案重建 Neo4j 知识图谱'

    def add_arguments(self, parser):
        parser.add_argument(
            '--teach-dir', type=str, default='/teach',
            help='教案目录路径 (默认: /teach)'
        )
        parser.add_argument(
            '--schema-only', action='store_true', default=False,
            help='仅重建知识结构 (Topic + PREREQUISITE_OF)，跳过题目关联'
        )
        parser.add_argument(
            '--full', action='store_true', default=False,
            help='清除所有节点重建 (含 User/Problem/Submission)，默认仅清除 Topic'
        )
        parser.add_argument(
            '--from-db', action='store_true', default=False,
            help='从数据库 lesson_plan 表读取教案 (而非 /teach 磁盘文件)'
        )
        parser.add_argument(
            '--incremental', action='store_true', default=False,
            help='增量模式：MERGE Topic 节点，不删除已有数据，可反复执行'
        )

    def handle(self, *args, **options):
        teach_dir = options['teach_dir']
        schema_only = options['schema_only']
        full = options['full']
        from_db = options['from_db']
        incremental = options['incremental']
        client = neo4j_client

        # =============================================
        # 1. 读取教案数据（数据库 or 磁盘文件）
        # =============================================
        if from_db:
            all_topics_data = read_lesson_plans_from_db()
            self.stdout.write(f'从数据库 lesson_plan 表读取 {len(all_topics_data)} 份教案')
        else:
            all_topics_data = parse_filenames_to_topics(teach_dir)
            self.stdout.write(f'从 {teach_dir} 读取 {len(all_topics_data)} 份教案')

        if not all_topics_data:
            self.stderr.write(self.style.ERROR('未找到任何教案'))
            return

        topic_names = set(td['name'] for td in all_topics_data)

        if incremental:
            # 增量模式：MERGE Topic，不删除已有数据
            self.stdout.write(self.style.WARNING('增量模式：MERGE Topic 节点（保留已有数据）...'))
            for td in all_topics_data:
                name = td['name']
                content = td.get('content') or read_file_content(td.get('filepath'))
                difficulty = parse_complexity_from_content(content)
                importance = parse_importance(name)
                client.run_query(
                    """
                    MERGE (t:Topic {name: $name})
                    SET t.difficulty = $difficulty,
                        t.importance = $importance,
                        t.source_file = $source_file
                    """,
                    {
                        'name': name,
                        'difficulty': str(difficulty),
                        'importance': str(importance),
                        'source_file': td['source_file'],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f'已 MERGE {len(all_topics_data)} 个 Topic 节点'))
        else:
            # 全量重建：清除旧 Topic 再 CREATE
            if full:
                self.stdout.write(self.style.WARNING('清除所有节点及关系...'))
                try:
                    client.run_query("MATCH (n) DETACH DELETE n")
                    self.stdout.write(self.style.SUCCESS('已清除所有节点及关系'))
                except Exception as e:
                    self.stderr.write(f'清除节点失败: {e}')
            else:
                self.stdout.write(self.style.WARNING('清除旧的 Topic 节点及相关关系...'))
                try:
                    client.run_query("MATCH (t:Topic) DETACH DELETE t")
                    self.stdout.write(self.style.SUCCESS('已清除所有 Topic 节点'))
                except Exception as e:
                    self.stderr.write(f'清除 Topic 节点失败: {e}')

            for td in all_topics_data:
                name = td['name']
                content = td.get('content') or read_file_content(td.get('filepath'))
                difficulty = parse_complexity_from_content(content)
                importance = parse_importance(name)
                client.run_query(
                    """
                    CREATE (t:Topic {
                        name: $name,
                        difficulty: $difficulty,
                        importance: $importance,
                        source_file: $source_file
                    })
                    """,
                    {
                        'name': name,
                        'difficulty': str(difficulty),
                        'importance': str(importance),
                        'source_file': td['filename'],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f'已创建 {len(all_topics_data)} 个 Topic 节点'))

        # =============================================
        # 3. 构建 PREREQUISITE_OF 关系
        # =============================================
        self.stdout.write('构建 PREREQUISITE_OF 关系...')

        prereq_created = 0
        succ_created = 0

        for td in all_topics_data:
            name = td['name']
            content = td.get('content') or read_file_content(td.get('filepath'))
            rels = parse_relationships(content)

            for prereq_name in rels['prerequisites']:
                matched = fuzzy_match_topic_name(prereq_name, topic_names)
                if matched and matched != name:
                    try:
                        client.run_query(
                            """
                            MATCH (t1:Topic {name: $prereq}), (t2:Topic {name: $name})
                            WHERE NOT EXISTS((t1)-[:PREREQUISITE_OF]->(t2))
                            CREATE (t1)-[:PREREQUISITE_OF]->(t2)
                            """,
                            {'prereq': matched, 'name': name}
                        )
                        prereq_created += 1
                    except Exception:
                        pass

            for succ_name in rels['successors']:
                matched = fuzzy_match_topic_name(succ_name, topic_names)
                if matched and matched != name:
                    try:
                        client.run_query(
                            """
                            MATCH (t1:Topic {name: $name}), (t2:Topic {name: $succ})
                            WHERE NOT EXISTS((t1)-[:PREREQUISITE_OF]->(t2))
                            CREATE (t1)-[:PREREQUISITE_OF]->(t2)
                            """,
                            {'name': name, 'succ': matched}
                        )
                        succ_created += 1
                    except Exception:
                        pass

        self.stdout.write(
            self.style.SUCCESS(
                f'创建 {prereq_created} 条前置关系, '
                f'{succ_created} 条后继关系'
            )
        )

        # =============================================
        # 4. 关联题目和知识点
        # =============================================
        if not schema_only:
            if full:
                self.stdout.write('从 PostgreSQL 重建 Problem 节点...')
                try:
                    from problem.models import Problem
                    problems = Problem.objects.filter(visible=True)
                    prob_count = 0
                    for p in problems:
                        client.run_query(
                            """
                            MERGE (:Problem {
                                problem_id: $id,
                                _id: $_id,
                                title: $title,
                                difficulty: $difficulty
                            })
                            """,
                            {
                                'id': p.id,
                                '_id': p._id,
                                'title': p.title,
                                'difficulty': p.difficulty,
                            }
                        )
                        prob_count += 1
                        if prob_count % 200 == 0:
                            self.stdout.write(f'  已处理 {prob_count} 道题目...')
                    self.stdout.write(
                        self.style.SUCCESS(f'已创建 {prob_count} 个 Problem 节点')
                    )
                except Exception as e:
                    self.stderr.write(f'重建 Problem 节点失败: {e}')

            self.stdout.write('关联题目和知识点...')

            try:
                from problem.models import ProblemTag, Problem

                all_tags = list(ProblemTag.objects.values_list('name', flat=True).distinct())

                existing = {
                    r['name'] for r in client.run_query(
                        "MATCH (t:Topic) RETURN t.name AS name"
                    )
                }

                relation_created = 0
                skipped = 0

                for tag_name in all_tags:
                    if not tag_name:
                        continue

                    if tag_name not in existing:
                        continue

                    try:
                        problem_ids = list(
                            Problem.objects
                            .filter(tags__name=tag_name)
                            .values_list('id', flat=True)
                        )
                    except Exception:
                        problem_ids = []

                    for pid in problem_ids:
                        try:
                            client.run_query(
                                """
                                MATCH (p:Problem {problem_id: $pid})
                                MATCH (t:Topic {name: $name})
                                WHERE NOT EXISTS((p)-[:BELONGS_TO]->(t))
                                CREATE (p)-[:BELONGS_TO]->(t)
                                """,
                                {'pid': pid, 'name': tag_name}
                            )
                            relation_created += 1
                        except Exception:
                            skipped += 1

                self.stdout.write(
                    self.style.SUCCESS(
                        f'关联 {relation_created} 条 BELONGS_TO 关系'
                        f'{", 跳过 " + str(skipped) + " 条" if skipped > 0 else ""}'
                    )
                )

            except Exception as e:
                self.stderr.write(f'关联题目和知识点失败: {e}')
        else:
            self.stdout.write(self.style.WARNING('--schema-only 模式, 跳过题目关联'))

        # =============================================
        # 5. 统计
        # =============================================
        topic_count = client.run_query(
            "MATCH (t:Topic) RETURN count(t) AS cnt"
        )[0]['cnt']
        prereq_count = client.run_query(
            "MATCH ()-[r:PREREQUISITE_OF]->() RETURN count(r) AS cnt"
        )[0]['cnt']
        belongs_count = client.run_query(
            "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS cnt"
        )[0]['cnt']

        self.stdout.write('')
        self.stdout.write('=' * 50)
        self.stdout.write(f'Topic 节点: {topic_count}')
        self.stdout.write(f'PREREQUISITE_OF 关系: {prereq_count}')
        self.stdout.write(f'BELONGS_TO 关系: {belongs_count}')
        self.stdout.write('=' * 50)
        self.stdout.write(
            self.style.SUCCESS('知识图谱重建完成!')
        )
