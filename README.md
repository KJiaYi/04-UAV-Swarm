# 无人机集群巡检模拟系统 README

# 无人机集群巡检模拟系统

## 项目简介

基于 MARL 的无人机集群协同巡检仿真系统，实现路径规划、通信模拟、任务覆盖与碰撞规避。

## 核心功能

- 无人机集群基础控制（起飞/移动/着陆）

- 可配置通信延迟/丢包模拟

- MADDPG 多智能体协同规划

- 任务点生成与覆盖检测

- 碰撞检测与规避

- 预训练数据集生成

## 快速开始

### 环境要求

|类型|要求|
|---|---|
|系统|Windows/Linux|
|依赖|AirSim、Python 3.7+|
|库|airsim、numpy、torch、pandas|
### 安装运行

1. 安装 AirSim：参考 [官方指南](https://microsoft.github.io/AirSim/installation/)

2. 安装依赖：`pip install airsim numpy torch pandas`

3. 启动 AirSim 模拟器（如 Blocks 环境）

4. 运行：
`python uav_swarm_patrol_demo.py # 主程序`
`python uav_data_generator.py # 生成数据集（可选）`

## 文件说明

|文件名称|核心功能说明|
|---|---|
|uav_communication_sim.py|负责无人机集群间的通信过程模拟，核心实现通信延迟（通过休眠机制）和丢包率（随机概率控制）功能，输出各无人机状态的传输结果，为协同决策提供通信层支撑|
|uav_swarm_patrol_demo.py|项目主程序，整合全流程逻辑：初始化AirSim客户端与无人机（起飞、定高）、生成并分配巡检任务点、加载MADDPG模型执行协同规划、实时检测碰撞与任务覆盖进度、输出巡检评估指标（覆盖率、碰撞次数等）|
|uav_maddpg_model.py|定义多智能体强化学习（MADDPG）核心模型，包含单无人机对应的Actor网络（输入状态输出动作）和全局共享的Critic网络（评估策略价值），集成优化器与奖励计算逻辑，支持模型训练与参数更新|
|uav_data_generator.py|生成无人机集群巡检的模拟数据集，包含任务点分布、无人机轨迹、状态动作等数据，整理后保存为CSV格式文件，用于MADDPG模型的预训练与性能优化|
> （注：文档部分内容可能由 AI 生成）