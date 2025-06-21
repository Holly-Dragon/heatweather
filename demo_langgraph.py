"""
LangGraph架构演示 - 展示工作流优势
简化版本，专注于演示LangGraph的工作流编排能力
"""

import asyncio
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END, START

# 定义状态
class DemoState(TypedDict):
    temperature: float
    step: int
    customer_decision: str
    rider_action: str
    government_policy: str
    platform_response: str
    messages: list

class LangGraphDemo:
    """LangGraph工作流演示"""
    
    def __init__(self):
        self.graph = self._build_demo_graph()
    
    def _build_demo_graph(self):
        """构建演示图"""
        
        def environment_node(state: DemoState) -> DemoState:
            """环境节点"""
            state["temperature"] = 42.5
            state["step"] += 1
            state["messages"].append(f"🌡️ 步骤{state['step']}: 环境更新 - 温度{state['temperature']}°C")
            return state
        
        def customer_node(state: DemoState) -> DemoState:
            """客户节点"""
            temp = state["temperature"]
            if temp > 40:
                decision = "担心骑手健康，暂不下单"
            else:
                decision = "正常下单"
            
            state["customer_decision"] = decision
            state["messages"].append(f"👤 步骤{state['step']}: 客户决策 - {decision}")
            return state
        
        def rider_node(state: DemoState) -> DemoState:
            """骑手节点"""
            temp = state["temperature"]
            if temp > 42:
                action = "温度过高，选择休息"
            else:
                action = "接单配送"
            
            state["rider_action"] = action
            state["messages"].append(f"🚴 步骤{state['step']}: 骑手行动 - {action}")
            return state
        
        def government_node(state: DemoState) -> DemoState:
            """政府节点"""
            temp = state["temperature"]
            if temp > 40:
                policy = f"启动高温预警，发放补贴{50 if temp > 42 else 30}元/人"
            else:
                policy = "正常监测"
            
            state["government_policy"] = policy
            state["messages"].append(f"🏛️ 步骤{state['step']}: 政府政策 - {policy}")
            return state
        
        def platform_node(state: DemoState) -> DemoState:
            """平台节点"""
            rider_action = state["rider_action"]
            if "休息" in rider_action:
                response = "检测到骑手大量休息，提高配送费用10%"
            else:
                response = "运营正常"
            
            state["platform_response"] = response
            state["messages"].append(f"💼 步骤{state['step']}: 平台响应 - {response}")
            return state
        
        def summary_node(state: DemoState) -> DemoState:
            """总结节点"""
            state["messages"].append("📊 本轮仿真完成")
            state["messages"].append(f"   - 环境: {state['temperature']}°C极端高温")
            state["messages"].append(f"   - 客户: {state['customer_decision']}")
            state["messages"].append(f"   - 骑手: {state['rider_action']}")
            state["messages"].append(f"   - 政府: {state['government_policy']}")
            state["messages"].append(f"   - 平台: {state['platform_response']}")
            return state
        
        # 构建图
        workflow = StateGraph(DemoState)
        
        # 添加节点
        workflow.add_node("environment", environment_node)
        workflow.add_node("customer", customer_node)
        workflow.add_node("rider", rider_node)
        workflow.add_node("government", government_node)
        workflow.add_node("platform", platform_node)
        workflow.add_node("summary", summary_node)
        
        # 设置边（执行顺序）
        workflow.add_edge(START, "environment")
        workflow.add_edge("environment", "customer")
        workflow.add_edge("customer", "rider")
        workflow.add_edge("rider", "government")
        workflow.add_edge("government", "platform")
        workflow.add_edge("platform", "summary")
        workflow.add_edge("summary", END)
        
        return workflow.compile()
    
    async def run_demo(self):
        """运行演示"""
        print("🤖 LangGraph工作流架构演示")
        print("="*50)
        
        # 初始状态
        initial_state = DemoState(
            temperature=0.0,
            step=0,
            customer_decision="",
            rider_action="",
            government_policy="",
            platform_response="",
            messages=[]
        )
        
        # 运行图
        final_state = await self.graph.ainvoke(initial_state)
        
        # 显示结果
        print("\n📋 执行流程:")
        for message in final_state["messages"]:
            print(f"  {message}")
        
        print("\n✨ LangGraph架构优势:")
        print("  ✅ 清晰的工作流定义")
        print("  ✅ 自动状态管理")
        print("  ✅ 异步并行支持")
        print("  ✅ 灵活的执行控制")
        print("  ✅ 可视化图结构")

def compare_architectures():
    """比较不同架构的优劣"""
    print("\n🔍 架构对比分析")
    print("="*50)
    
    print("📊 传统循环架构 vs LangGraph架构")
    print()
    
    print("🔄 传统方式:")
    print("  • 简单for循环，逐个调用Agent")
    print("  • 硬编码执行顺序")
    print("  • 状态管理复杂")
    print("  • 难以并行化")
    print("  • 扩展性差")
    
    print("\n🚀 LangGraph方式:")
    print("  • 图结构定义工作流")
    print("  • 自动状态传递")
    print("  • 支持条件分支")
    print("  • 内置并行支持")
    print("  • 高度可扩展")
    
    print("\n💡 LLM集成优势:")
    print("  • 智能决策推理")
    print("  • 自然语言交互")
    print("  • 可解释的AI行为")
    print("  • 适应复杂场景")

async def main():
    """主演示程序"""
    print("🌟 LangGraph + LLM 多主体仿真系统演示")
    print("="*60)
    
    # 运行LangGraph演示
    demo = LangGraphDemo()
    await demo.run_demo()
    
    # 架构对比
    compare_architectures()
    
    print("\n" + "="*60)
    print("🎯 完整仿真系统特性:")
    print("  🌡️ 极端高温环境模拟")
    print("  🤖 DeepSeek LLM智能决策")
    print("  🔄 LangGraph工作流编排")
    print("  📊 多维度数据分析")
    print("  📈 实时可视化图表")
    print("  💾 详细日志记录")
    
    print("\n🚀 运行完整仿真:")
    print("  python langgraph_simulation.py")
    print()
    print("🧪 测试LLM集成:")
    print("  python test_llm_integration.py")

if __name__ == "__main__":
    asyncio.run(main())
