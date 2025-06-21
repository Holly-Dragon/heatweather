"""
基于LangGraph的多主体仿真系统
模拟极端高温下政府措施对外卖小哥幸福感和健康程度的影响
"""

from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import random

from environment import Environment
from agents import Customer, Rider, Platform, Government, Order
from utils import SimulationLogger, print_agent_workflow

class SimulationState(TypedDict):
    """仿真状态定义"""
    environment: Environment
    customers: List[Customer]
    riders: List[Rider]
    platform: Platform
    government: Government
    orders: List[Order]
    current_orders: List[Order]  # 当前未完成订单
    logger: SimulationLogger
    day: int
    hour: int
    simulation_running: bool

class HeatWeatherSimulation:
    """极端高温仿真主类"""
    
    def __init__(self, num_customers: int = 10, num_riders: int = 3, simulation_days: int = 30):
        self.num_customers = num_customers
        self.num_riders = num_riders
        self.simulation_days = simulation_days
        
        # 初始化环境和Agent
        self.environment = Environment()
        self.customers = [Customer(f"customer_{i}") for i in range(num_customers)]
        self.riders = [Rider(f"rider_{i}") for i in range(num_riders)]
        self.platform = Platform()
        self.government = Government()
        self.logger = SimulationLogger()
        
        # 初始化订单列表
        self.all_orders = []
        self.current_orders = []
        
        # 构建LangGraph
        self.graph = self._build_simulation_graph()
        
    def _build_simulation_graph(self):
        """构建仿真图"""
        
        def environment_step(state: SimulationState) -> SimulationState:
            """环境步骤 - 推进时间"""
            env = state["environment"]
            env.advance_hour()
            
            state["day"] = env.current_day
            state["hour"] = env.current_hour
            
            # 检查是否结束仿真
            if env.current_day >= self.simulation_days:
                state["simulation_running"] = False
            
            return state
        
        def customer_workflow(state: SimulationState) -> SimulationState:
            """客户工作流"""
            if not state["environment"].is_meal_time():
                return state
                
            env_state = state["environment"].get_environment_state()
            
            for customer in state["customers"]:
                # Observation
                obs = customer.observe(env_state)
                
                # Thought
                thought = customer.think(env_state)
                
                # Action - 决定是否下单
                order = customer.decide_order(env_state)
                if order:
                    state["current_orders"].append(order)
                    state["orders"].append(order)
                
                # 记录日志
                action = f"下单: {order.order_id[:8]}" if order else "未下单"
                state["logger"].log_agent_action(
                    state["day"], state["hour"], "Customer", customer.agent_id,
                    obs, thought, action
                )
                
                print_agent_workflow(customer, "Customer Workflow")
            
            return state
        
        def rider_workflow(state: SimulationState) -> SimulationState:
            """骑手工作流"""
            env_state = state["environment"].get_environment_state()
            available_orders = [o for o in state["current_orders"] if not o.rider_id]
            
            for rider in state["riders"]:
                if not rider.on_duty:
                    continue
                    
                # Observation
                obs = rider.observe(env_state, available_orders)
                
                # Thought
                thought = rider.think(env_state, available_orders)
                
                # Action
                action_type = rider.decide_action(env_state, available_orders)
                action_desc = ""
                
                if action_type == "deliver" and available_orders:
                    # 选择一个订单配送
                    order = random.choice(available_orders)
                    result = rider.deliver_order(order, env_state)
                    available_orders.remove(order)
                    
                    # 客户评分和小费
                    customer = next(c for c in state["customers"] if c.agent_id == order.customer_id)
                    rating = customer.rate_order(order, rider.health)
                    tip = customer.decide_tip(order, rider.health)
                    
                    action_desc = f"配送订单 {order.order_id[:8]}，收入{result['income']:.1f}元"
                    
                elif action_type == "rest":
                    rider.rest(env_state)
                    action_desc = "休息恢复"
                    
                elif action_type == "complain":
                    complaint = rider.complain(env_state)
                    action_desc = "投诉工作条件"
                    
                else:
                    action_desc = "等待"
                
                # 记录日志
                state["logger"].log_agent_action(
                    state["day"], state["hour"], "Rider", rider.agent_id,
                    obs, thought, action_desc
                )
                
                print_agent_workflow(rider, "Rider Workflow")
            
            return state
        
        def platform_workflow(state: SimulationState) -> SimulationState:
            """平台工作流"""
            if state["hour"] != 23:  # 每天结束时执行
                return state
                
            platform = state["platform"]
            
            # Observation
            obs = platform.observe(state["riders"], state["orders"])
            
            # Thought
            thought = platform.think(state["riders"], state["environment"].get_environment_state())
            
            # Actions
            actions = []
            
            # 计算收益
            profit = platform.calc_profit(state["orders"])
            actions.append(f"计算收益: {profit:.1f}元")
            
            # 考虑解雇表现差的骑手
            for rider in state["riders"]:
                if platform.consider_fire_rider(rider):
                    actions.append(f"解雇骑手: {rider.agent_id}")
            
            # 每周缴税
            if state["day"] % 7 == 0:
                tax = platform.pay_tax(state["government"])
                actions.append(f"缴税: {tax:.1f}元")
            
            action_desc = "; ".join(actions)
            
            # 记录日志
            state["logger"].log_agent_action(
                state["day"], state["hour"], "Platform", platform.agent_id,
                obs, thought, action_desc
            )
            
            print_agent_workflow(platform, "Platform Workflow")
            
            return state
        
        def government_workflow(state: SimulationState) -> SimulationState:
            """政府工作流"""
            if state["hour"] != 22:  # 每天晚上22点执行
                return state
                
            government = state["government"]
            env_state = state["environment"].get_environment_state()
            
            # Observation
            obs = government.observe(env_state, state["riders"])
            
            # Thought
            thought = government.think(env_state, state["riders"])
            
            # Actions
            actions = []
            
            # 高温补贴决策
            subsidy = government.decide_subsidy(env_state, state["riders"])
            if subsidy > 0:
                actions.append(f"发放高温补贴: {subsidy:.1f}元")
            
            # 增设纳凉点决策
            if government.decide_build_shelter(env_state, state["riders"]):
                # 第二天生效
                state["environment"].add_shelter(0.1)  # 增加10%覆盖率
                actions.append("决定增设纳凉点")
            
            action_desc = "; ".join(actions) if actions else "保持观察"
            
            # 记录日志
            state["logger"].log_agent_action(
                state["day"], state["hour"], "Government", government.agent_id,
                obs, thought, action_desc
            )
            
            print_agent_workflow(government, "Government Workflow")
            
            return state
        
        def daily_summary(state: SimulationState) -> SimulationState:
            """每日总结"""
            if state["hour"] == 0 and state["day"] > 0:  # 新的一天开始时总结前一天
                prev_day = state["day"] - 1
                state["logger"].log_daily_stats(
                    prev_day, state["riders"], state["customers"], 
                    state["orders"], state["environment"].get_environment_state(),
                    state["government"], state["platform"]
                )
                state["logger"].print_daily_summary(prev_day)
                
                # 重置平台日收益
                state["platform"].daily_revenue = 0.0
                # 重置骑手日收益
                for rider in state["riders"]:
                    rider.daily_income = 0.0
            
            return state
        
        def check_simulation_end(state: SimulationState) -> str:
            """检查仿真是否结束"""
            if state["simulation_running"]:
                return "continue_simulation"
            else:
                return "end_simulation"
        
        # 构建图
        workflow = StateGraph(SimulationState)
        
        # 添加节点
        workflow.add_node("environment_step", RunnableLambda(environment_step))
        workflow.add_node("customer_workflow", RunnableLambda(customer_workflow))
        workflow.add_node("rider_workflow", RunnableLambda(rider_workflow))
        workflow.add_node("platform_workflow", RunnableLambda(platform_workflow))
        workflow.add_node("government_workflow", RunnableLambda(government_workflow))
        workflow.add_node("daily_summary", RunnableLambda(daily_summary))
        
        # 添加边
        workflow.set_entry_point("environment_step")
        workflow.add_edge("environment_step", "daily_summary")
        workflow.add_edge("daily_summary", "customer_workflow")
        workflow.add_edge("customer_workflow", "rider_workflow")
        workflow.add_edge("rider_workflow", "platform_workflow")
        workflow.add_edge("platform_workflow", "government_workflow")
        
        # 条件边 - 检查是否继续仿真
        workflow.add_conditional_edges(
            "government_workflow",
            check_simulation_end,
            {
                "continue_simulation": "environment_step",
                "end_simulation": END
            }
        )
        
        return workflow.compile()
    
    def run_simulation(self) -> Dict[str, Any]:
        """运行仿真"""
        print("开始极端高温下外卖配送多主体仿真...")
        print(f"仿真参数: {self.num_customers}个客户, {self.num_riders}个骑手, {self.simulation_days}天")
        
        # 初始状态
        initial_state = SimulationState(
            environment=self.environment,
            customers=self.customers,
            riders=self.riders,
            platform=self.platform,
            government=self.government,
            orders=self.all_orders,
            current_orders=self.current_orders,
            logger=self.logger,
            day=0,
            hour=6,
            simulation_running=True
        )
        
        # 运行仿真
        try:
            final_state = self.graph.invoke(initial_state)
            
            print("\n=== 仿真完成 ===")
            print("正在生成分析报告...")
            
            # 保存日志
            self.logger.save_logs(f"simulation_logs_day{self.simulation_days}.json")
            
            # 生成图表
            self.logger.plot_simulation_results()
            self.logger.plot_rider_analysis()
            
            # 最终统计
            final_stats = self._generate_final_stats_from_logger()
            return final_stats
            
        except Exception as e:
            print(f"仿真过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_final_stats_from_logger(self) -> Dict[str, Any]:
        """从日志生成最终统计数据"""
        if not self.logger.daily_stats:
            return {}
            
        last_day_stats = self.logger.daily_stats[-1]
        
        stats = {
            "simulation_days": self.simulation_days,
            "total_orders": last_day_stats.get("total_orders", 0),
            "completed_orders": last_day_stats.get("completed_orders", 0),
            "completion_rate": (last_day_stats.get("completed_orders", 0) / 
                              max(1, last_day_stats.get("total_orders", 1))),
            "avg_final_health": last_day_stats.get("avg_rider_health", 0),
            "avg_final_happiness": last_day_stats.get("avg_rider_happiness", 0),
            "total_complaints": last_day_stats.get("total_complaints", 0),
            "total_subsidies": last_day_stats.get("government_subsidies", 0),
            "shelters_built": last_day_stats.get("shelters_built", 0),
            "final_shelter_rate": last_day_stats.get("shelter_rate", 0)
        }
        
        print("\n=== 最终统计结果 ===")
        print(f"总仿真天数: {stats['simulation_days']}")
        print(f"总订单数: {stats['total_orders']}")
        print(f"完成订单数: {stats['completed_orders']}")
        print(f"订单完成率: {stats['completion_rate']:.2%}")
        print(f"骑手平均最终健康: {stats['avg_final_health']:.1f}/10")
        print(f"骑手平均最终幸福感: {stats['avg_final_happiness']:.1f}/10")
        print(f"总投诉数: {stats['total_complaints']}")
        print(f"政府总补贴: {stats['total_subsidies']:.1f}元")
        print(f"新建纳凉点: {stats['shelters_built']}个")
        print(f"最终阴凉覆盖率: {stats['final_shelter_rate']:.2f}")
        
        return stats
