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

1. 安装 AirSim：参考官方指南

2. 安装依赖：`pip install airsim numpy torch pandas`

3. 启动 AirSim 模拟器（如 Blocks 环境）

4. 运行：
`python uav_swarm_patrol_demo.py # 主程序`
`python uav_data_generator.py # 生成数据集（可选）`

## 文件说明

|文件|功能|
|---|---|
|uav_communication_sim.py|通信延迟/丢包模拟|
|uav_swarm_patrol_demo.py|主程序（初始化/巡检/训练）|
|uav_maddpg_model.py|MADDPG 模型定义|
|uav_data_generator.py|预训练数据集生成|
