"""
基于LangGraph的多主体仿真系统
集成DeepSeek LLM进行智能决策
"""

from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
import random
import asyncio
from datetime import datetime

from environment import Environment
from llm_agents import LLMCustomer, LLMRider, LLMGovernment, LLMPlatform
from utils import SimulationLogger
from llm_config import check_llm_status

class LangGraphSimulationState(TypedDict):
    """LangGraph仿真状态定义"""
    # 环境和Agent
    environment: Environment
    customers: List[LLMCustomer]
    riders: List[LLMRider]
    platform: LLMPlatform
    government: LLMGovernment
    
    # 仿真控制
    current_day: int
    current_hour: int
    simulation_days: int
    
    # 数据记录
    all_orders: List[Any]
    current_orders: List[Any]
    logger: SimulationLogger
    
    # 状态标志
    simulation_running: bool
    step_count: int

class LangGraphHeatWeatherSimulation:
    """基于LangGraph的极端高温仿真系统"""
    
    def __init__(self, num_customers: int = 5, num_riders: int = 2, simulation_days: int = 5):
        self.num_customers = num_customers
        self.num_riders = num_riders
        self.simulation_days = simulation_days
        
        # 检查LLM状态
        check_llm_status()
        
        # 初始化组件
        self.environment = Environment()
        self.customers = [LLMCustomer(f"customer_{i}") for i in range(num_customers)]
        self.riders = [LLMRider(f"rider_{i}") for i in range(num_riders)]
        self.platform = LLMPlatform()
        self.government = LLMGovernment()
        self.logger = SimulationLogger()
        
        # 构建LangGraph
        self.graph = self._build_simulation_graph()
        
        print(f"🚀 LangGraph仿真系统初始化完成")
        print(f"📊 配置: {num_customers}个客户, {num_riders}个骑手, {simulation_days}天")
        
    def _build_simulation_graph(self):
        """构建LangGraph仿真图"""
        
        def environment_step(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """环境推进步骤"""
            env = state["environment"]
            env.advance_hour()
            
            state["current_day"] = env.current_day
            state["current_hour"] = env.current_hour
            state["step_count"] += 1
            
            # 检查仿真结束条件
            if env.current_day >= state["simulation_days"]:
                state["simulation_running"] = False
                print(f"🏁 仿真完成: 总共{state['step_count']}步")
            
            return state
        
        def customer_workflow(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """客户工作流节点"""
            env_state = state["environment"].get_environment_state()
            
            # 只在用餐时间执行客户行为
            if not env_state["is_meal_time"]:
                return state
            
            print(f"🍽️ {state['current_hour']:02d}:00 客户决策时间")
            
            for customer in state["customers"]:
                try:
                    order = customer.observe_and_decide(env_state)
                    if order:
                        state["current_orders"].append(order)
                        state["all_orders"].append(order)
                        print(f"  📱 {customer.agent_id}: 下单 {order.cost:.0f}元")
                        
                        # 记录到日志
                        state["logger"].log_agent_action(
                            state["current_day"], state["current_hour"], 
                            "Customer", customer.agent_id,
                            customer.observations[-1] if customer.observations else "",
                            customer.thoughts[-1] if customer.thoughts else "",
                            customer.actions[-1] if customer.actions else ""
                        )
                except Exception as e:
                    print(f"❌ 客户{customer.agent_id}决策失败: {e}")
            
            return state
        
        def rider_workflow(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """骑手工作流节点"""
            env_state = state["environment"].get_environment_state()
            available_orders = [o for o in state["current_orders"] if not hasattr(o, 'rider_id')]
            
            temp = env_state['temperature']
            if temp > 42:
                print(f"🔥 {state['current_hour']:02d}:00 极端高温警报! {temp:.1f}°C")
            elif temp > 38:
                print(f"🌡️ {state['current_hour']:02d}:00 高温预警 {temp:.1f}°C")
            
            # 处理每个骑手的决策
            for rider in state["riders"]:
                if not rider.on_duty:
                    continue
                
                try:
                    action = rider.observe_and_decide(env_state, available_orders)
                    
                    if action == "deliver" and available_orders:
                        # 选择并配送订单
                        order = random.choice(available_orders)
                        result = rider.deliver_order(order, env_state)
                        available_orders.remove(order)
                        state["current_orders"].remove(order)
                        
                        # 客户评分和小费
                        customer = next(c for c in state["customers"] if c.agent_id == order.customer_id)
                        rating = customer.rate_order(order, rider.health)
                        tip = customer.decide_tip(order, rider.health)
                        
                        print(f"  🚴 {rider.agent_id}: 配送完成 +{result['income']:.0f}元 健康{rider.health:.1f}/10")
                        
                        # 健康警报
                        if rider.health < 3:
                            print(f"    ⚠️ {rider.agent_id} 健康状况危险!")
                        elif rider.health < 5:
                            print(f"    💔 {rider.agent_id} 健康状况较差")
                            
                    elif action == "rest":
                        rider.rest(env_state)
                        print(f"  💤 {rider.agent_id}: 休息恢复")
                        
                    elif action == "complain":
                        complaint = rider.complain(env_state)
                        print(f"  📢 {rider.agent_id}: 投诉工作条件")
                    
                    # 记录到日志
                    state["logger"].log_agent_action(
                        state["current_day"], state["current_hour"],
                        "Rider", rider.agent_id,
                        rider.observations[-1] if rider.observations else "",
                        rider.thoughts[-1] if rider.thoughts else "",
                        rider.actions[-1] if rider.actions else ""
                    )
                    
                except Exception as e:
                    print(f"❌ 骑手{rider.agent_id}决策失败: {e}")
            
            return state
        
        def platform_workflow(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """平台工作流节点"""
            # 只在每天结束时执行平台决策
            if state["current_hour"] != 23:
                return state
            
            print(f"💼 平台运营决策")
            
            try:
                platform = state["platform"]
                actions = platform.observe_and_decide(state["riders"], state["all_orders"])
                
                # 计算日收益
                profit = platform.calc_profit(state["all_orders"])
                print(f"  💰 日收益: {profit:.0f}元")
                
                # 缴税
                if state["current_day"] % 7 == 0:  # 每周缴税
                    tax = platform.pay_tax(state["government"])
                    print(f"  💸 缴税: {tax:.0f}元")
                
                # 记录到日志
                state["logger"].log_agent_action(
                    state["current_day"], state["current_hour"],
                    "Platform", platform.agent_id,
                    platform.observations[-1] if platform.observations else "",
                    platform.thoughts[-1] if platform.thoughts else "",
                    platform.actions[-1] if platform.actions else ""
                )
                
            except Exception as e:
                print(f"❌ 平台决策失败: {e}")
            
            return state
        
        def government_workflow(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """政府工作流节点"""
            # 只在每天晚上22点执行政府决策
            if state["current_hour"] != 22:
                return state
            
            print(f"🏛️ 政府政策决策")
            
            try:
                government = state["government"]
                env_state = state["environment"].get_environment_state()
                policies = government.observe_and_decide(env_state, state["riders"])
                
                # 执行政策
                if policies["subsidy"] > 0:
                    print(f"  🎁 发放高温补贴: {policies['subsidy']:.0f}元")
                
                if policies["shelter"]:
                    # 增加纳凉点覆盖率
                    state["environment"].add_shelter(0.1)
                    print(f"  🏠 增设纳凉点，覆盖率提升至{state['environment'].shelter_rate:.2f}")
                
                # 记录到日志
                state["logger"].log_agent_action(
                    state["current_day"], state["current_hour"],
                    "Government", government.agent_id,
                    government.observations[-1] if government.observations else "",
                    government.thoughts[-1] if government.thoughts else "",
                    government.actions[-1] if government.actions else ""
                )
                
            except Exception as e:
                print(f"❌ 政府决策失败: {e}")
            
            return state
        
        def daily_summary(state: LangGraphSimulationState) -> LangGraphSimulationState:
            """每日总结节点"""
            # 在新的一天开始时总结前一天
            if state["current_hour"] == 0 and state["current_day"] > 0:
                prev_day = state["current_day"] - 1
                print(f"\n📊 Day {prev_day + 1} 总结")
                
                # 记录每日统计
                state["logger"].log_daily_stats(
                    prev_day, state["riders"], state["customers"],
                    state["all_orders"], state["environment"].get_environment_state(),
                    state["government"], state["platform"]
                )
                
                # 打印简要统计
                completed_orders = len([o for o in state["all_orders"] if hasattr(o, 'delivered') and o.delivered])
                avg_health = sum(r.health for r in state["riders"]) / len(state["riders"])
                total_complaints = sum(len(r.complaints) for r in state["riders"])
                
                print(f"  📦 完成订单: {completed_orders}")
                print(f"  🏥 骑手平均健康: {avg_health:.1f}/10")
                print(f"  📢 总投诉数: {total_complaints}")
                print(f"  🏛️ 政府补贴: {state['government'].subsidies_paid:.0f}元")
                print(f"  🏠 纳凉点覆盖率: {state['environment'].shelter_rate:.2f}")
                
                # 重置日统计
                state["platform"].daily_revenue = 0.0
                for rider in state["riders"]:
                    rider.daily_income = 0.0
            
            return state
        
        def check_continuation(state: LangGraphSimulationState) -> str:
            """检查是否继续仿真"""
            if state["simulation_running"]:
                return "continue"
            else:
                return "end"
        
        # 构建图
        workflow = StateGraph(LangGraphSimulationState)
        
        # 添加节点
        workflow.add_node("environment_step", environment_step)
        workflow.add_node("daily_summary", daily_summary)
        workflow.add_node("customer_workflow", customer_workflow)
        workflow.add_node("rider_workflow", rider_workflow)
        workflow.add_node("platform_workflow", platform_workflow)
        workflow.add_node("government_workflow", government_workflow)
        
        # 设置入口点
        workflow.add_edge(START, "environment_step")
        
        # 添加边（执行顺序）
        workflow.add_edge("environment_step", "daily_summary")
        workflow.add_edge("daily_summary", "customer_workflow")
        workflow.add_edge("customer_workflow", "rider_workflow")
        workflow.add_edge("rider_workflow", "platform_workflow")
        workflow.add_edge("platform_workflow", "government_workflow")
        
        # 条件边：检查是否继续
        workflow.add_conditional_edges(
            "government_workflow",
            check_continuation,
            {
                "continue": "environment_step",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def run_simulation(self) -> Dict[str, Any]:
        """运行仿真"""
        print("\n" + "="*60)
        print("🌡️ 基于LangGraph的极端高温多主体仿真")
        print("🤖 集成DeepSeek LLM智能决策")
        print("="*60)
        
        # 初始状态
        initial_state = LangGraphSimulationState(
            environment=self.environment,
            customers=self.customers,
            riders=self.riders,
            platform=self.platform,
            government=self.government,
            current_day=0,
            current_hour=6,  # 从早上6点开始
            simulation_days=self.simulation_days,
            all_orders=[],
            current_orders=[],
            logger=self.logger,
            simulation_running=True,
            step_count=0
        )
        
        try:
            # 运行LangGraph
            print(f"🚀 开始仿真...")
            final_state = await self.graph.ainvoke(initial_state)
            
            print("\n🎉 仿真完成!")
            
            # 生成最终报告
            self._generate_final_report(final_state)
            
            return self._extract_results(final_state)
            
        except Exception as e:
            print(f"\n❌ 仿真失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_final_report(self, final_state: Dict[str, Any]):
        """生成最终报告"""
        print("\n" + "="*60)
        print("📈 最终仿真报告")
        print("="*60)
        
        # 保存详细日志
        log_filename = f"langgraph_simulation_{self.simulation_days}days.json"
        self.logger.save_logs(log_filename)
        
        # 基础统计
        riders = final_state["riders"]
        orders = final_state["all_orders"]
        government = final_state["government"]
        platform = final_state["platform"]
        
        total_orders = len(orders)
        completed_orders = len([o for o in orders if hasattr(o, 'delivered') and o.delivered])
        completion_rate = completed_orders / max(1, total_orders)
        
        avg_health = sum(r.health for r in riders) / len(riders)
        avg_happiness = sum(r.happiness for r in riders) / len(riders)
        total_complaints = sum(len(r.complaints) for r in riders)
        active_riders = len([r for r in riders if r.on_duty])
        
        print(f"📊 基础统计:")
        print(f"  - 仿真天数: {self.simulation_days}")
        print(f"  - 总订单数: {total_orders}")
        print(f"  - 完成订单: {completed_orders}")
        print(f"  - 完成率: {completion_rate:.1%}")
        print(f"  - 在职骑手: {active_riders}/{self.num_riders}")
        
        print(f"\n🏥 健康与幸福感:")
        print(f"  - 骑手平均健康: {avg_health:.1f}/10")
        print(f"  - 骑手平均幸福感: {avg_happiness:.1f}/10")
        print(f"  - 总投诉数: {total_complaints}")
        
        print(f"\n🏛️ 政府措施效果:")
        print(f"  - 总补贴支出: {government.subsidies_paid:.0f}元")
        print(f"  - 新建纳凉点: {government.shelters_built}个")
        print(f"  - 最终覆盖率: {final_state['environment'].shelter_rate:.2f}")
        
        print(f"\n💰 平台运营:")
        print(f"  - 平台总资金: {platform.cash:.0f}元")
        
        # LLM决策统计
        llm_decisions = 0
        rule_decisions = 0
        
        for agent_list in [riders, self.customers, [government, platform]]:
            for agent in agent_list:
                if hasattr(agent, 'thoughts'):
                    llm_decisions += len([t for t in agent.thoughts if "LLM" in t])
                    rule_decisions += len([t for t in agent.thoughts if "规则" in t])
        
        print(f"\n🤖 决策方式统计:")
        print(f"  - LLM智能决策: {llm_decisions}次")
        print(f"  - 规则基础决策: {rule_decisions}次")
        
        # 效果评估
        print(f"\n🎯 政策效果评估:")
        
        if avg_health >= 7:
            health_grade = "优秀 ✅"
        elif avg_health >= 5:
            health_grade = "良好 👍"
        elif avg_health >= 3:
            health_grade = "一般 ⚠️"
        else:
            health_grade = "较差 ❌"
        print(f"  - 健康水平: {health_grade}")
        
        if avg_happiness >= 7:
            happiness_grade = "优秀 😊"
        elif avg_happiness >= 5:
            happiness_grade = "良好 🙂"
        elif avg_happiness >= 3:
            happiness_grade = "一般 😐"
        else:
            happiness_grade = "较差 😟"
        print(f"  - 幸福感: {happiness_grade}")
        
        # 建议
        print(f"\n💡 AI分析建议:")
        if avg_health < 5:
            print("  📌 骑手健康堪忧，建议增加补贴和休息时间")
        if total_complaints > self.simulation_days * 2:
            print("  📌 投诉过多，需要改善工作环境")
        if completion_rate < 0.8:
            print("  📌 服务完成率偏低，需平衡效率与健康")
        if final_state['environment'].shelter_rate < 0.5:
            print("  📌 纳凉点覆盖率仍需提升")
        if active_riders < self.num_riders * 0.8:
            print("  📌 骑手流失严重，需改善待遇")
        
        print(f"\n💾 详细日志: {log_filename}")
        
        # 尝试生成图表
        try:
            self.logger.plot_simulation_results()
            print("📊 图表已生成: simulation_results.png")
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
    
    def _extract_results(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """提取仿真结果"""
        riders = final_state["riders"]
        orders = final_state["all_orders"]
        
        return {
            "simulation_days": self.simulation_days,
            "total_orders": len(orders),
            "completed_orders": len([o for o in orders if hasattr(o, 'delivered') and o.delivered]),
            "completion_rate": len([o for o in orders if hasattr(o, 'delivered') and o.delivered]) / max(1, len(orders)),
            "avg_health": sum(r.health for r in riders) / len(riders),
            "avg_happiness": sum(r.happiness for r in riders) / len(riders),
            "total_complaints": sum(len(r.complaints) for r in riders),
            "subsidies_paid": final_state["government"].subsidies_paid,
            "shelters_built": final_state["government"].shelters_built,
            "final_shelter_rate": final_state["environment"].shelter_rate,
            "platform_cash": final_state["platform"].cash
        }

def main():
    """主函数"""
    print("🤖 基于LangGraph和DeepSeek LLM的极端高温仿真系统")
    
    try:
        customers = int(input("客户数量 (默认5): ") or "5")
        riders = int(input("骑手数量 (默认2): ") or "2") 
        days = int(input("仿真天数 (默认5): ") or "5")
    except ValueError:
        customers, riders, days = 5, 2, 5
        print("使用默认参数")
    
    async def run():
        simulation = LangGraphHeatWeatherSimulation(customers, riders, days)
        results = await simulation.run_simulation()
        return results
    
    # 运行异步仿真
    results = asyncio.run(run())
    
    if results:
        print(f"\n✅ 仿真成功完成")
        print(f"🎯 订单完成率: {results['completion_rate']:.1%}")
        print(f"🏥 平均健康: {results['avg_health']:.1f}/10")
        print(f"😊 平均幸福感: {results['avg_happiness']:.1f}/10")
    else:
        print("\n❌ 仿真失败")

if __name__ == "__main__":
    main()
