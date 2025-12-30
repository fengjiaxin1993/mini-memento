import logging
from typing import List
from openai import OpenAI
import settings
from memory.episodic import EpisodicMemory
from memory.vector_store import MementoVectorStore

logger = logging.getLogger(__name__)


def format_memories_for_prompt(memories: List[dict]) -> str:
    parts = []
    for m in memories:
        if not m.get("is_success") and m.get("feedback"):
            parts.append(
                f"- 任务：“{m['task']}”\n"
                f"  你曾回答：“{m['output']}”，这是错误的。\n"
                f"  正确做法：“{m['feedback']}”\n"
            )
        elif m.get("is_success"):
            parts.append(
                f"- 成功案例：“{m['task']}” → “{m['output']}”"
            )
    return "\n".join(parts) if parts else "无相关历史经验。"


class MementoAgent:
    def __init__(self):
        self.llm_client = OpenAI(
            base_url=settings.api_base_url,  # Ollama API 地址
            api_key="ollama"  # Ollama 默认无需真实 API Key，填任意值即可
        )
        self.memory = MementoVectorStore(embed_model=settings.embed_model)

    def run(self, task: str) -> str:
        # Step 1: 检索相关历史记忆
        past_memories = self.memory.retrieve(task, k=2)

        # Step 2: 构建增强 prompt
        memory_context = format_memories_for_prompt(past_memories)
        prompt = f"""
你是一个智能助手。请参考以下过往经验：

{memory_context}

现在，请完成任务：
{task}
        """.strip()

        # Step 3: 调用 LLM
        response = self.llm_client.chat.completions.create(
            model=settings.llm_model,  # 指定模型
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature,  # 控制生成多样性
            max_tokens=settings.max_tokens  # 最大生成 token 数
        )
        logger.info(f"prompt:\n{prompt}")
        print(f"prompt:\n{prompt}")
        content = response.choices[0].message.content
        return content

    def log_experience(self, task: str, output: str, feedback: str = None, success: bool = False):
        """用户反馈后调用此方法记录经验"""
        mem = EpisodicMemory(
            task=task,
            output=output,
            feedback=feedback,
            is_success=success
        )
        self.memory.add_memory(mem)
