from dataclasses import dataclass
from typing import List, Optional


# 情景记忆模块
@dataclass
class EpisodicMemory:
    task: str  # 用户原始任务
    output: str  # 模型原始输出
    feedback: Optional[str] = None  # 用户纠正或正确答案
    is_success: bool = False  # 是否成功
