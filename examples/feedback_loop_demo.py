from agent import MementoAgent

agent = MementoAgent()

# 第一次问答（模型出错）
# task = "电力监控系统的十六字方针是什么？"
# # response = agent.run(task)
# response = "电力监控系统应具备以下十六字方针：实时诊断、优化运行、灵活控制和安全稳定。"
# print("模型回答:", response)  # 电力监控系统应具备以下十六字方针：实时诊断、优化运行、灵活控制和安全稳定。
#
# 用户反馈
# correct_answer = "电力监控系统的十六字方针是 “安全分区、网络专用、横向隔离、纵向认证”"
# agent.log_experience(
#     task=task,
#     output=response,
#     feedback=correct_answer,
#     success=False
# )
#
# 第二次类似任务
new_task = "电力监控系统的十六字方针是什么？"
new_response = agent.run(new_task)
print("修正后回答:", new_response)
# 输出将包含对闰年规则的正确应用！
