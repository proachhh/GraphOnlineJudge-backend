import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CHROMADB_DATA_DIR = os.environ.get(
    'CHROMADB_DATA_DIR',
    '/home/proach/OnlineJudgeDeploy/data/chromadb'
)


class VectorStoreService:
    def __init__(self, collection_name: str = 'oj_documents', persist_dir: str = None):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or CHROMADB_DATA_DIR
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_dir, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            existing = self._client.list_collections()
            existing_names = [c.name for c in existing] if existing else []

            if self.collection_name in existing_names:
                self._collection = self._client.get_collection(self.collection_name)
                logger.info(f"Connected to existing ChromaDB collection: '{self.collection_name}' "
                            f"at {self.persist_dir}")
            else:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={'hnsw:space': 'cosine'},
                )
                logger.info(f"Created new ChromaDB collection: '{self.collection_name}' "
                            f"at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None
            self._collection = None

    def add_documents(self, texts: List[str], sources: List[str] = None,
                      tags: List[str] = None) -> bool:
        if self._collection is None:
            logger.error("ChromaDB collection not initialized")
            return False

        if not texts:
            return True

        n = len(texts)
        ids = [f"doc_{i}" for i in range(self._collection.count(), self._collection.count() + n)]
        metadatas = []
        for i in range(n):
            meta = {}
            if sources and i < len(sources):
                meta['source'] = sources[i]
            if tags and i < len(tags):
                meta['tags'] = tags[i]
            metadatas.append(meta)

        try:
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Added {n} documents to ChromaDB collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if self._collection is None:
            logger.error("ChromaDB collection not initialized")
            return []

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
            )

            documents = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    item = {'content': doc, 'score': 0.0}
                    if results.get('distances') and results['distances'][0]:
                        item['score'] = round(1.0 - results['distances'][0][i], 4)
                    if results.get('metadatas') and results['metadatas'][0]:
                        item['metadata'] = results['metadatas'][0][i]
                    if results.get('ids') and results['ids'][0]:
                        item['id'] = results['ids'][0][i]
                    documents.append(item)

            return documents
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def delete_collection(self):
        if self._client is not None and self._collection is not None:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = None
                logger.info(f"Deleted ChromaDB collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Failed to delete ChromaDB collection: {e}")

    def count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def delete_by_source(self, source: str) -> int:
        if self._collection is None:
            return 0
        try:
            results = self._collection.get(
                where={'source': source},
                include=['metadatas'],
            )
            ids_to_delete = results.get('ids', [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents with source='{source}'")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete documents by source '{source}': {e}")
            return 0

    @property
    def is_ready(self) -> bool:
        return self._collection is not None


_vector_store_instance = None


def get_vector_store(collection_name: str = 'oj_documents') -> VectorStoreService:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService(collection_name=collection_name)
    return _vector_store_instance
