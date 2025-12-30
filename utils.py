import logging
from typing import List

import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

import settings

logger = logging.getLogger(__name__)


# 先默认设置ollama方式，之后在扩展
# "api_base_url": "http://127.0.0.1:11434/v1",
def get_Embeddings(
        api_base_url: str = settings.api_base_url,
        embed_model: str = None
) -> OllamaEmbeddings:
    embed_model = embed_model
    try:
        return OllamaEmbeddings(
            base_url=api_base_url.replace("/v1", ""),
            model=embed_model,
        )
    except Exception as e:
        logger.exception(f"failed to create Embeddings for model: {embed_model}.")


# 向量归一化
def l2_normalize(vector: List[float]) -> List[float]:
    """
    对向量进行L2归一化（核心函数）
    :param vector: 原始向量（Ollama生成）
    :return: 归一化后的单位向量
    """
    # 转换为numpy数组便于计算
    vec_np = np.array(vector, dtype=np.float32)
    # 计算向量模长
    norm = np.linalg.norm(vec_np)
    # 避免除以0（模长为0时返回原向量）
    if norm < 1e-5:
        return vector
    # 归一化并转回列表
    normalized_vec = vec_np / norm
    return normalized_vec.tolist()


