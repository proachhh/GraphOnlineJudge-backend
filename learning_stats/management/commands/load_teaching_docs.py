import os
import re

from django.core.management.base import BaseCommand
from utils.vector_store import VectorStoreService


def strip_number_prefix(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r'^\d+[.\-\s_]*', '', name)
    name = re.sub(r'[_\-\s]+', ' ', name)
    return name.strip()


def split_by_headings(text: str, min_chunk_len: int = 50) -> list:
    if not text or not text.strip():
        return []

    text = text.strip()
    h2_sections = re.split(r'\n(?=## [^\n])', text)

    if len(h2_sections) > 2:
        chunks = []
        for section in h2_sections:
            section = section.strip()
            if len(section) >= min_chunk_len:
                chunks.append(section)
        return chunks

    h3_sections = re.split(r'\n(?=### [^\n])', h2_sections[0])
    if len(h3_sections) > 1:
        chunks = []
        for section in h3_sections[1:]:
            section = section.strip()
            if len(section) >= min_chunk_len:
                chunks.append(section)
        return chunks

    if len(h2_sections) >= 2 and len(h2_sections[1].strip()) >= min_chunk_len:
        return [h2_sections[1].strip()]

    return [text] if len(text) >= min_chunk_len else []


class Command(BaseCommand):
    help = '将 .md 教案导入 ChromaDB 向量数据库'

    def add_arguments(self, parser):
        parser.add_argument(
            'directory', type=str, nargs='?',
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', '..', 'teaching_docs'),
            help='教案目录路径'
        )
        parser.add_argument(
            '--collection', type=str, default='oj_documents',
            help='ChromaDB 集合名称 (默认: oj_documents)'
        )
        parser.add_argument(
            '--persist-dir', type=str, default=None,
            help='ChromaDB 持久化目录 (默认: 环境变量 CHROMADB_DATA_DIR)'
        )

    def handle(self, *args, **options):
        directory = options['directory']
        collection_name = options['collection']
        persist_dir = options['persist_dir']

        if not os.path.isdir(directory):
            self.stderr.write(self.style.ERROR(f"目录不存在: {directory}"))
            return

        files = sorted([
            f for f in os.listdir(directory)
            if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))
        ])

        if not files:
            self.stdout.write(f"目录 {directory} 下没有找到 .md 文件")
            return

        store = VectorStoreService(collection_name=collection_name, persist_dir=persist_dir)

        if not store.is_ready:
            self.stderr.write(self.style.ERROR(
                "ChromaDB 未就绪，请先安装 chromadb 并确保存储目录可写"
            ))
            return

        total_files = len(files)
        total_chunks = 0
        total_inserted = 0

        for filename in files:
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                self.stdout.write(f"  跳过空文件: {filename}")
                continue

            chunks = split_by_headings(content)

            if not chunks:
                self.stdout.write(f"  跳过(无有效片段): {filename}")
                continue

            tag_name = strip_number_prefix(filename)

            deleted = store.delete_by_source(filename)
            if deleted > 0:
                self.stdout.write(f"  已清除旧片段: {filename} ({deleted} 条)")

            texts = []
            sources = []
            tags = []
            for chunk in chunks:
                texts.append(chunk)
                sources.append(filename)
                tags.append(tag_name)

            ok = store.add_documents(texts=texts, sources=sources, tags=tags)

            if ok:
                total_inserted += len(chunks)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ {filename}: {len(chunks)} 个片段 → 标签: \"{tag_name}\""
                    )
                )
            else:
                self.stderr.write(
                    self.style.ERROR(f"  ✗ {filename}: 入库失败")
                )

            total_chunks += len(chunks)

        self.stdout.write("")
        self.stdout.write("=" * 50)
        self.stdout.write(f"处理文件数: {total_files}")
        self.stdout.write(f"生成片段数: {total_chunks}")
        self.stdout.write(f"入库成功数: {total_inserted}")
        self.stdout.write(f"集合总文档数: {store.count()}")
        self.stdout.write("=" * 50)
