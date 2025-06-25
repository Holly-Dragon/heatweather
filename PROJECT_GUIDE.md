# 项目指南

## 1. 项目结构

```
heatweather/
├── langgraph_simulation.py # 仿真主程序入口
├── llm_agents.py           # 定义基于LLM的Agent决策逻辑
├── agents.py               # 定义Agent的基础属性和行为
├── environment.py          # 环境模拟器
├── llm_config.py           # LLM模型配置
├── utils.py                # 工具函数和日志记录
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明文档
```

## 2. 核心模块说明

*   **`langgraph_simulation.py`**: 仿真运行的主入口。它初始化所有Agent和环境，并按时间步长驱动整个仿真过程。最终的仿真结果（如图表和日志）也由该文件生成。

*   **`llm_agents.py`**: 定义了各个Agent（如骑手、政府）如何使用大语言模型（LLM）进行决策。每个Agent的决策逻辑都在这里实现。

*   **`agents.py`**: 定义了Agent的基础类，包括其属性（如健康值、收入）和基本行为（如移动、接单）。

*   **`environment.py`**: 模拟外部环境，包括生成温度变化、管理地图信息（纳凉点位置）和生成外卖订单。

*   **`llm_config.py`**: 配置LLM的API Key和模型参数。**在运行前，请务必在此文件中填入你的DeepSeek API Key。**

*   **`utils.py`**: 包含一些辅助功能，如日志记录、数据处理等。

## 3. 运行步骤

1.  **安装依赖**: 
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置API Key**: 
    打开 `llm_config.py` 文件，在指定位置填入你的DeepSeek API Key。

3.  **运行仿真**: 
    ```bash
    python langgraph_simulation.py
    ```

4.  **查看结果**: 
    仿真结束后，程序会自动生成并展示结果图表，同时会在项目根目录下保存 `simulation_log.json` 文件，记录了详细的仿真过程数据。
