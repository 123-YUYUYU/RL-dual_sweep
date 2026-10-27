# 模块接口与扩展

## 环境调用

```python
from scheduling.config import load_config
from scheduling.environment import EdgeSchedulingEnv

env = EdgeSchedulingEnv(load_config("configs/default.json"))
observation, reset_info = env.reset(seed=7)
done = False
while not done:
    action = env.encode(edge_budget=env.edges, preference=1, model_rank=0)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(env.summary())
```

环境提供与常见 RL 工具一致的五返回值交互形式，但当前不是 Gymnasium 的子类，也不声明 observation/action space 对象。连接第三方 RL 框架时应添加适配器并正确表达有限时域终止。

## 替换工作负载

在 `scenario.py` 中替换 `generate_trace`，保持返回的任务列表、信道矩阵及后台负载矩阵长度与 horizon 一致。场景生成必须独立于策略，保证配对评估可比。任务 ID 在一个 episode 内应唯一，deadline 是绝对时隙结束边界。

## 替换外层策略

保留 `encode/decode` 即可接入其他离散策略。`DQN.values` 只接收当前观测；监督模块提供 `suggest(observation)` 接口。若改为逐任务动作，需要同步修改解码、任务分配、模型兼容检查、状态输入和模型维度校验，再重新训练。

## 替换内层求解器

`allocate` 输入当前任务、资源容量与信道，输出与任务列表对齐的 frequency、power、rate 数组以及资源违反量、迭代次数、目标变化与收敛状态。上传阶段和计算阶段互斥，不应为上传未完成的任务分配有效计算进度。替换方法也应保持每节点算力预算与每设备总发射功率约束。

## 增加实验

在独立输出目录中保存配置与日志。扩展任务等级、队列纪律、节点规模或负载分布时，同步增加相应约束和状态不变量测试，记录参数来源与评估口径。
