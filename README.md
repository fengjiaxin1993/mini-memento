# mini-memento
mini版本的只记录历史案例，不微调模型，让大模型回答的越来越好

# 1. 项目简介
通过外接向量库，记录和大模型交互的记录，如果错误则进行更正，根据历史相关问答调整提示词，不断修改prompt
chromadb
openai
主要使用上面2个包


## 1.python项目环境配置

### 1.1 环境安装
conda create -n mini-memento python=3.11
conda activate mini-memento
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
or
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

### 1.2.运行代码
cd examples
python feedback_loop_demo.py