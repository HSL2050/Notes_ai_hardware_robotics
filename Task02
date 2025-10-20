# 手眼标定（Hand-Eye Calibration）学习笔记

## 1. 手眼标定的定义

手眼标定（Hand-Eye Calibration）是机器人视觉系统中关键的一步，其目标是确定相机（Eye）与机械臂末端执行器（Hand）之间的空间关系。

换言之，就是求解**相机坐标系**与**机械臂末端坐标系**之间的刚性变换矩阵。

手眼标定通常有两种形式：

- **Eye-in-Hand（手上眼）**：相机安装在机械臂末端执行器上。
- **Eye-to-Hand（手外眼）**：相机固定在外部，观察机械臂的工作区域。

---

## 2. 手眼标定的数学模型

### 2.1 坐标变换回顾

一个坐标系 \( \{A\} \) 到 \( \{B\} \) 的刚性变换可表示为：

\[
{}^{A}T_{B} =
\begin{bmatrix}
R_{A}^{B} & t_{A}^{B} \\
0 & 1
\end{bmatrix}
\]

其中：
- \( R_{A}^{B} \in SO(3) \)：旋转矩阵  
- \( t_{A}^{B} \in \mathbb{R}^3 \)：平移向量  

---

### 2.2 Eye-in-Hand 模型推导

设：
- \( {}^{B}T_{E} \)：机械臂基坐标系到末端执行器坐标系的变换  
- \( {}^{E}T_{C} \)：末端执行器到相机的变换（**待求**）  
- \( {}^{C}T_{O} \)：相机到目标坐标系的变换  
- \( {}^{B}T_{O} \)：基坐标系到目标的变换  

则有：

\[
{}^{B}T_{O} = {}^{B}T_{E} \, {}^{E}T_{C} \, {}^{C}T_{O}
\]

如果相机观察的目标是固定的（即 \( {}^{B}T_{O} \) 不变），则在两次不同的机械臂位姿 \( i \) 和 \( j \) 下：

\[
{}^{B}T_{E_i} \, {}^{E}T_{C} \, {}^{C_i}T_{O} = {}^{B}T_{E_j} \, {}^{E}T_{C} \, {}^{C_j}T_{O}
\]

移项可得：

\[
({}^{B}T_{E_j})^{-1} \, {}^{B}T_{E_i} \, {}^{E}T_{C} = {}^{E}T_{C} \, {}^{C_j}T_{O} \, ({}^{C_i}T_{O})^{-1}
\]

记：

\[
A_i = ({}^{B}T_{E_j})^{-1} \, {}^{B}T_{E_i}, \quad
B_i = {}^{C_j}T_{O} \, ({}^{C_i}T_{O})^{-1}
\]

得到**经典手眼标定方程：**

\[
A_i X = X B_i
\]

其中 \( X = {}^{E}T_{C} \) 为待求的手眼变换。

---

### 2.3 分解为旋转和平移方程

设：

\[
A_i =
\begin{bmatrix}
R_{A_i} & t_{A_i} \\
0 & 1
\end{bmatrix},
\quad
B_i =
\begin{bmatrix}
R_{B_i} & t_{B_i} \\
0 & 1
\end{bmatrix},
\quad
X =
\begin{bmatrix}
R_X & t_X \\
0 & 1
\end{bmatrix}
\]

代入 \( A_i X = X B_i \)，可分为两个方程：

#### (1) 旋转部分：

\[
R_{A_i} R_X = R_X R_{B_i}
\]

#### (2) 平移部分：

\[
R_{A_i} t_X + t_{A_i} = R_X t_{B_i} + t_X
\]

---

### 2.4 旋转部分求解

对于多个运动对 \((A_i, B_i)\)，可写为：

\[
R_{A_i} R_X = R_X R_{B_i}, \quad i = 1, 2, \dots, n
\]

常用求解方法包括：

- **Tsai–Lenz 算法 (1989)**  
- **Park–Martin 算法 (1994)**  
- **Dual Quaternion 方法**

#### Tsai–Lenz 旋转解法思路：

利用旋转轴-角参数化：
\[
R = \exp(\hat{\omega}\theta)
\]

线性化后可通过最小二乘解得旋转向量 \(\omega_X\)。

---

### 2.5 平移部分求解

已知 \( R_X \)，平移方程为：

\[
(R_{A_i} - I)t_X = R_X t_{B_i} - t_{A_i}
\]

将多组数据叠加成线性系统，可用最小二乘求解 \( t_X \)。

---

## 3. Eye-to-Hand 模型

当相机固定在外部时（Eye-to-Hand），设：

- \( {}^{C}T_{B} \)：相机到机械臂基坐标系的变换（**待求**）

则有：

\[
{}^{C}T_{O} = {}^{C}T_{B} \, {}^{B}T_{E} \, {}^{E}T_{O}
\]

同理可推导出：

\[
A_i X = X B_i
\]

此时 \( X = {}^{C}T_{B} \)，只是几何意义不同。

---

## 4. 求解算法概述

| 算法 | 特点 | 是否线性 | 精度 | 是否常用 |
|------|------|-----------|-------|-----------|
| Tsai–Lenz | 经典算法，分步求解旋转和平移 | 半线性 | 高 | ✅ |
| Park–Martin | 基于李代数最小化误差 | 非线性 | 高 | ✅ |
| Dual Quaternion | 同时求旋转和平移 | 线性 | 中等 | ✅ |
| Nonlinear Optimization | 最小化 \(\| A_i X - X B_i \|\) | 非线性 | 最高 | ⚙️需迭代 |

---

## 5. 实验步骤（实际操作流程）

1. **标定相机内参**：获取相机的内在参数矩阵 \(K\)。  
2. **采集数据**：让机械臂多次移动末端执行器，拍摄标定板。  
3. **计算 \(A_i, B_i\)**：  
   - \(A_i\)：由机械臂的正运动学得到。  
   - \(B_i\)：由相机图像中标定板位姿估计得到。  
4. **求解 \(X\)**：使用上述算法（Tsai–Lenz 或优化法）。  
5. **验证结果**：检查重投影误差或坐标变换一致性。

---

## 6. 总结与思考

- 手眼标定的核心是求解矩阵方程 \( A_i X = X B_i \)，反映了**相机与机械臂之间的刚性关系**。  
- 实际误差主要来源于：
  - 相机位姿估计误差
  - 机械臂末端位姿误差
  - 数据分布不均衡
- 改进方向：
  - 使用更多视角、多样化运动
  - 优化求解（非线性最小化）
  - 联合标定（相机内参 + 手眼一起）

---

## 7. 参考文献

1. Tsai, R.Y. and Lenz, R.K., *A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration*, IEEE Journal of Robotics and Automation, 1989.  
2. Park, F.C. and Martin, B.J., *Robot Sensor Calibration: Solving AX = XB on the Euclidean Group*, IEEE Transactions on Robotics and Automation, 1994.  
3. Daniilidis, K., *Hand-Eye Calibration Using Dual Quaternions*, The International Journal of Robotics Research, 1999.

---
