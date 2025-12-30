import os
from typing import List
import chromadb

from memory.episodic import EpisodicMemory
from settings import MEMORY_DB_PATH
from utils import get_Embeddings, l2_normalize


def get_vs_path(vector_name: str):
    return os.path.join(MEMORY_DB_PATH, vector_name)


class MementoVectorStore:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.embed_model = embed_model
        self.kb_name = "memory_db"
        self.kb_path = get_vs_path(self.embed_model)
        self.client = chromadb.PersistentClient(path=self.kb_path)
        self.collection = self.client.get_or_create_collection(
            name=self.kb_name,
            metadata={
                "description": "基于余弦相似度的向量集合",
                "hnsw:space": "cosine"  # 指定索引距离度量为余弦相似度（核心参数）
                # 可选：调整HNSW索引参数（影响性能和精度）
                # "hnsw:m": 16,  # 每个节点的邻居数，默认16
                # "hnsw:ef_construction": 100  # 构建索引时的探索深度，默认100
            }
        )
        self.embed_func = get_Embeddings(embed_model=self.embed_model)

    def add_memory(self, mem: EpisodicMemory):
        text = f"{mem.task} {mem.output}"
        emb = self.embed_func.embed_documents(texts=[text])[0]
        normalize_emb = l2_normalize(emb)

        self.collection.add(
            embeddings=[normalize_emb],
            documents=[text],
            metadatas=[{
                "task": mem.task,
                "output": mem.output,
                "feedback": mem.feedback,
                "is_success": mem.is_success
            }],
            ids=[f"mem_{hash(text)}"]
        )

    def retrieve(self, query: str, k=2) -> List[dict]:
        emb = self.embed_func.embed_documents(texts=[query])[0]
        normalize_emb = l2_normalize(emb)
        results = self.collection.query(
            query_embeddings=[normalize_emb],
            n_results=k,
            include=["metadatas"]
        )
        return results["metadatas"][0]


if __name__ == "__main__":
    embed_model = "quentinz/bge-small-zh-v1.5"
    vs = MementoVectorStore(embed_model)
    # em1 = EpisodicMemory(task="测试", output="你好", feedback="", is_success=True)
    # em2 = EpisodicMemory(task="正式", output="你好", feedback="不好", is_success=False)
    # vs.add_memory(em1)
    # vs.add_memory(em2)
    res = vs.retrieve("正式", k=2)
    print(res)
