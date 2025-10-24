
# 强化学习与智能决策学习笔记

> 聚焦强化学习（Reinforcement Learning, RL）及其在机器人控制中的应用，重点理解 **PPO（Proximal Policy Optimization）算法**。

---

## 一、强化学习与智能决策概述

### 1.1 强化学习的基本框架

强化学习研究智能体（Agent）如何通过与环境（Environment）交互来最大化长期收益。
在每个时间步，智能体：

* 观察状态 $s_t$
* 选择动作 $a_t$
* 获得奖励 $r_t$
* 进入新状态 $s_{t+1}$

智能体的目标是最大化期望累计回报：

$$
\max_\pi ; \mathbb{E}*{\pi} \Big[ \sum*{t=0}^\infty \gamma^t r_t \Big]
$$

其中：

* $\pi(a|s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率；
* $\gamma \in (0,1)$ 是折扣因子，平衡短期与长期收益。

---

### 1.2 智能决策的本质

强化学习的本质是实现 **感知 → 决策 → 执行 → 反馈** 的闭环优化过程：

| 环节 | 含义           | 示例             |          |
| -- | ------------ | -------------- | -------- |
| 感知 | 获取状态 $s_t$   | 摄像头、激光雷达、位置传感器 |          |
| 决策 | 根据策略 $\pi(a  | s)$ 选择动作       | 抓取、转向、移动 |
| 执行 | 动作作用于环境      | 机械臂运动、无人机飞行    |          |
| 反馈 | 环境返回奖励 $r_t$ | 成功与否、能量损耗等     |          |

---

## 二、机器人强化学习框架

### 2.1 机器人强化学习的挑战

与虚拟环境相比，机器人强化学习面临：

* **交互成本高**：物理试错代价大
* **状态与动作连续高维**：难以离散化
* **动力学不确定**：存在噪声、摩擦、延迟
* **仿真与现实差距大（Sim-to-Real）**

常见应对方式：

* 仿真训练 + 实机微调
* 域随机化（Domain Randomization）
* 模型预测控制（MPC）与强化学习结合

---

## 三、PPO算法原理

### 3.1 策略梯度基础

强化学习的优化目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}*{\pi*\theta}\left[\sum_t \gamma^t r_t\right]
$$

通过策略梯度方法直接优化策略参数 $\theta$，其梯度形式为：

$$
\nabla_\theta J(\theta) =
\mathbb{E}*{\pi*\theta} \big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\big]
$$

其中：

* $A^{\pi_\theta}(s,a)$ 是优势函数（Advantage Function），表示动作优于平均水平的程度。

---

### 3.2 TRPO（Trust Region Policy Optimization）

为避免更新过大导致性能退化，TRPO 引入 KL 散度约束：

$$
\max_\theta ; \mathbb{E} \Big[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s,a)
\Big]
\quad \text{s.t.} \quad
D_{KL}(\pi_{\theta_{\text{old}}} \Vert \pi_\theta) \le \delta
$$

但 TRPO 的计算较复杂，不易实现。

---

### 3.3 PPO（Proximal Policy Optimization）

PPO 是对 TRPO 的简化改进，通过**截断概率比（clipped ratio）**实现近端优化。

定义：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

构造目标函数：

$$
L^{clip}(\theta) =
\mathbb{E}_t \Big[
\min\big(
r_t(\theta) A_t,
; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\big)
\Big]
$$

* 当新旧策略差距过大时（超过 $\epsilon$），使用截断限制梯度更新；
* 从而保持更新稳定，防止策略崩溃。

完整的 PPO 损失函数通常为：

$$
L^{PPO} = L^{clip}(\theta)

* c_1 (V_\theta(s_t) - V_t^{target})^2

- c_2 S[\pi_\theta](s_t)
  $$

其中：

* $V_\theta$：价值函数；
* $S[\pi_\theta]$：策略熵（鼓励探索）；
* $c_1, c_2$：损失权重。

---

## 四、PPO在机器人控制中的应用

### 4.1 常见场景

| 应用      | 状态输入      | 动作输出    | 奖励设计        |
| ------- | --------- | ------- | ----------- |
| 机械臂抓取   | 关节角度、目标位置 | 关节速度/力矩 | 成功抓取 + 能量惩罚 |
| 双足机器人   | 姿态角、角速度   | 关节力矩    | 行走距离 + 稳定性  |
| 无人机控制   | 姿态、角速度    | 推力、角速度  | 路径跟踪误差      |
| 轮式机器人导航 | 激光雷达、位置   | 线速度、角速度 | 距离目标减少量     |

---

### 4.2 PPO 控制机器人示例（伪代码）

```python
env = RobotEnv()
agent = PPO(state_dim=env.state_dim, action_dim=env.action_dim)

for episode in range(max_episodes):
    state = env.reset()
    for t in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    agent.update()
```

训练完成后可将策略迁移到真实机器人上，配合域随机化或模仿学习提高稳定性。

---

### 4.3 优点与局限

**优点：**

* 更新稳定（clip机制防止崩溃）
* 收敛速度快
* 适合连续动作控制任务

**局限：**

* 奖励函数设计依赖经验
* 仿真与现实存在差距
* 样本效率不如模型辅助方法

---

## 五、总结与展望

* 强化学习实现了机器人从“感知—决策—控制”的闭环智能。
* PPO以其稳定性和高效性成为机器人控制的主流算法。
* 未来发展方向包括：

  * 模型驱动强化学习（Model-based RL）
  * 模仿学习与RL融合
  * 多模态感知与具身智能（Embodied AI）
  * 真实世界强化学习（Real-World RL）

---

## 参考资料

1. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.
2. Sutton & Barto, *Reinforcement Learning: An Introduction (2nd Edition)*.
3. OpenAI, *Spinning Up in Deep RL*.
4. Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, ICLR 2016.
