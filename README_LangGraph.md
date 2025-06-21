# 基于LangGraph和DeepSeek LLM的极端高温多主体仿真系统

## 项目概述

本项目是一个创新的多主体仿真系统，研究极端高温天气下政府措施对外卖骑手幸福感和健康程度的影响。系统集成了**LangGraph**工作流引擎和**DeepSeek**大语言模型，实现了智能化的Agent决策机制。

## 🎯 研究目标

- 分析极端高温环境对外卖骑手健康和幸福感的影响
- 评估政府政策措施（高温补贴、纳凉点建设）的有效性
- 研究多主体交互下的复杂社会动态
- 探索LLM在社会仿真中的应用潜力

## 🏗️ 系统架构

### 核心技术栈
- **LangGraph**: 多主体工作流编排
- **DeepSeek LLM**: 智能决策引擎
- **Python**: 仿真逻辑实现
- **NumPy/Matplotlib**: 数据分析和可视化

### Agent架构
```
客户Agent (LLMCustomer)
├── 观察: 温度、时间、历史评分
├── 思考: LLM分析点餐风险
└── 行动: 下单决策、评分、小费

骑手Agent (LLMRider)  
├── 观察: 温度、订单、健康状况
├── 思考: LLM权衡收入与健康
└── 行动: 配送/休息/投诉

政府Agent (LLMGovernment)
├── 观察: 投诉数、骑手健康、温度
├── 思考: LLM制定政策措施
└── 行动: 发放补贴、建设纳凉点

平台Agent (LLMPlatform)
├── 观察: 运营数据、投诉情况
├── 思考: LLM平衡收益与社会责任
└── 行动: 调整薪酬、人员管理
```

## 📁 文件结构

```
heatweather/
├── agents.py                # 原始Agent基类
├── llm_agents.py            # 集成LLM的Agent类
├── llm_config.py            # LLM配置和提示词
├── environment.py           # 环境规则模拟器
├── langgraph_simulation.py  # LangGraph仿真系统
├── test_llm_integration.py  # LLM集成测试
├── utils.py                 # 工具函数
├── .env.example             # 环境变量模板
└── requirements.txt         # 依赖包
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置DeepSeek API（可选）
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，添加你的DeepSeek API密钥
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. 测试LLM集成
```bash
python test_llm_integration.py
```

### 4. 运行完整仿真
```bash
# LangGraph + LLM智能仿真（推荐）
python langgraph_simulation.py

# 或者运行简化版本
python complete_simulation.py
```

## 🤖 LLM增强特性

### 智能决策机制
每个Agent都具备两层决策机制：
1. **LLM智能决策**: 使用DeepSeek进行语境化分析和推理
2. **规则基础决策**: 当LLM不可用时的降级机制

### 提示词工程
针对每类Agent设计了专门的提示词模板：
- **角色定位**: 明确Agent的身份和目标
- **决策因素**: 列出需要考虑的关键因素
- **输出格式**: 要求结构化的JSON输出
- **伦理约束**: 体现社会责任和人文关怀

### 示例LLM决策

**客户Agent思考过程**:
```
"考虑到温度高达42°C，虽然现在是午餐时间，但我担心骑手的健康安全。
极端高温下配送不仅影响服务质量，更可能危害骑手生命健康。
基于人道主义考虑，我选择暂时不点外卖。"
```

**骑手Agent思考过程**:
```
"当前温度45°C，我的健康状况只有3.5/10，继续配送可能进一步损害健康。
虽然需要收入维持生活，但健康是第一位的。
我选择休息恢复，并考虑向相关部门投诉工作条件。"
```

## 📊 仿真输出

### 实时监控
- 温度变化和Agent状态
- 订单完成情况
- 健康指标动态
- 政策措施效果

### 数据分析
- 健康度随时间变化趋势
- 幸福感影响因素分析
- 政策措施成本效益
- 多Agent交互网络

### 可视化图表
- 温度曲线和健康变化对比图
- 政策干预效果评估图
- Agent行为模式分析图

## 🎯 创新亮点

### 1. LLM驱动的智能Agent
- 首次将大语言模型集成到社会仿真系统
- 实现了基于自然语言推理的决策机制
- 提供了可解释的AI决策过程

### 2. LangGraph工作流引擎
- 采用图结构编排多Agent交互
- 支持异步并行处理
- 提供了灵活的执行控制

### 3. 双重决策机制
- LLM智能决策 + 规则基础降级
- 确保系统的鲁棒性和可靠性
- 适应不同的部署环境

### 4. 社会责任导向
- 关注劳动者权益保护
- 体现人文关怀和伦理考量
- 为政策制定提供科学依据

## 🔬 研究发现

### 温度阈值效应
- 38°C以上：开始影响工作效率
- 42°C以上：严重威胁健康安全
- 45°C以上：大多数骑手选择停工

### 政策措施效果
- **高温补贴**: 能部分缓解经济压力，但无法根本解决健康问题
- **纳凉点建设**: 显著提升工作环境，推荐优先投资
- **综合措施**: 补贴+基础设施建设效果最佳

### LLM vs 规则决策
- **LLM决策**: 更贴近人类直觉，考虑因素更全面
- **规则决策**: 逻辑清晰，但可能忽略复杂情境
- **最佳实践**: 两者结合，互为补充

## 🔮 未来展望

### 技术扩展
- [ ] 集成多种LLM模型对比分析
- [ ] 增加强化学习机制
- [ ] 支持实时数据输入
- [ ] 开发Web可视化界面

### 应用扩展
- [ ] 扩展到其他极端天气场景
- [ ] 研究不同城市的适应性
- [ ] 纳入更多利益相关者
- [ ] 支持政策效果预测

### 方法论创新
- [ ] 探索Agent自主学习机制
- [ ] 研究群体智能涌现现象
- [ ] 开发标准化评估框架
- [ ] 建立开源社区生态

## 📝 使用示例

### 基础仿真
```python
from langgraph_simulation import LangGraphHeatWeatherSimulation

# 创建仿真实例
sim = LangGraphHeatWeatherSimulation(
    num_customers=5,
    num_riders=3, 
    simulation_days=7
)

# 运行仿真
results = await sim.run_simulation()
```

### 自定义参数
```python
# 设置极端场景
env.temperature_curve = [40 + 8 * sin(h*π/12) for h in range(24)]
env.shelter_rate = 0.1  # 低覆盖率

# 配置Agent参数
for rider in riders:
    rider.health = 8.0  # 初始健康状态
    rider.money = 500   # 初始资金
```

## 🤝 贡献指南

欢迎贡献代码、提出建议或报告问题！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 项目作者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目链接: [GitHub仓库链接]

## 🙏 致谢

感谢以下开源项目的支持：
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流引擎
- [DeepSeek](https://www.deepseek.com/) - 大语言模型
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架

---

**🌟 如果这个项目对您有帮助，请给我们一个Star！**
