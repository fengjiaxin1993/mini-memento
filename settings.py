import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

MEMORY_DB_PATH = os.path.join(ROOT_PATH, "data", "memory_db")

# 大模型相关信息
api_base_url: str = "http://127.0.0.1:11434/v1"
llm_model: str = "qwen2.5:0.5b"
embed_model: str = "quentinz/bge-small-zh-v1.5"
temperature: float = 0.9  # 控制生成多样性
max_tokens: int = 512  # 最大生成 token 数
