"""
简化版本的仿真测试
用于验证系统功能
"""

import sys
import os
import random
import numpy as np
from typing import Dict, List

# 简化版的仿真，不使用LangGraph，直接按步骤执行
from environment import Environment
from agents import Customer, Rider, Platform, Government, Order
from utils import SimulationLogger, print_agent_workflow

class SimpleHeatWeatherSimulation:
    """简化版极端高温仿真"""
    
    def __init__(self, num_customers: int = 3, num_riders: int = 2, simulation_days: int = 3):
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
        
    def run_simple_simulation(self):
        """运行简化仿真"""
        print("开始简化版极端高温仿真...")
        print(f"配置: {self.num_customers}个客户, {self.num_riders}个骑手, {self.simulation_days}天")
        
        for day in range(self.simulation_days):
            print(f"\n=== Day {day + 1} ===")
            
            # 每天24小时
            for hour in range(24):
                self.environment.current_day = day
                self.environment.current_hour = hour
                
                env_state = self.environment.get_environment_state()
                
                print(f"\nHour {hour}: 温度 {env_state['temperature']:.1f}°C")
                
                # 客户行为 - 在用餐时间
                if env_state["is_meal_time"]:
                    self._customer_workflow(env_state)
                
                # 骑手行为 - 每小时都可能行动
                self._rider_workflow(env_state)
                
                # 平台行为 - 每天结束时
                if hour == 23:
                    self._platform_workflow(env_state)
                
                # 政府行为 - 每天晚上22点
                if hour == 22:
                    self._government_workflow(env_state)
                
                self.environment.advance_hour()
            
            # 每日统计
            final_env_state = self.environment.get_environment_state()
            self.logger.log_daily_stats(
                day, self.riders, self.customers, self.all_orders,
                final_env_state, self.government, self.platform
            )
            self.logger.print_daily_summary(day)
        
        # 最终报告
        print("\n=== 仿真完成 ===")
        self._generate_final_report()
        
        # 保存结果
        self.logger.save_logs("simple_simulation_logs.json")
        print("日志已保存至 simple_simulation_logs.json")
        
        return True
    
    def _customer_workflow(self, env_state: Dict):
        """客户工作流"""
        for customer in self.customers:
            # Observation
            obs = customer.observe(env_state)
            
            # Thought
            thought = customer.think(env_state)
            
            # Action
            order = customer.decide_order(env_state)
            if order:
                self.current_orders.append(order)
                self.all_orders.append(order)
                print(f"  客户{customer.agent_id}下单: {order.cost:.1f}元, {order.distance:.1f}km")
            
            # 记录
            action = f"下单: {order.order_id[:8]}" if order else "未下单"
            self.logger.log_agent_action(
                env_state["day"], env_state["hour"], "Customer", customer.agent_id,
                obs, thought, action
            )
    
    def _rider_workflow(self, env_state: Dict):
        """骑手工作流"""
        available_orders = [o for o in self.current_orders if not o.rider_id]
        
        for rider in self.riders:
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
                order = random.choice(available_orders)
                result = rider.deliver_order(order, env_state)
                available_orders.remove(order)
                self.current_orders.remove(order)
                
                # 客户评分
                customer = next(c for c in self.customers if c.agent_id == order.customer_id)
                rating = customer.rate_order(order, rider.health)
                tip = customer.decide_tip(order, rider.health)
                
                action_desc = f"配送完成，收入{result['income']:.1f}元，评分{rating}星"
                print(f"  骑手{rider.agent_id}: {action_desc}")
                
            elif action_type == "rest":
                rider.rest(env_state)
                action_desc = "休息恢复"
                
            elif action_type == "complain":
                complaint = rider.complain(env_state)
                action_desc = "投诉工作条件"
                print(f"  骑手{rider.agent_id}投诉: 健康{rider.health:.1f}, 温度{env_state['temperature']:.1f}°C")
                
            # 记录
            self.logger.log_agent_action(
                env_state["day"], env_state["hour"], "Rider", rider.agent_id,
                obs, thought, action_desc
            )
    
    def _platform_workflow(self, env_state: Dict):
        """平台工作流"""
        # Observation
        obs = self.platform.observe(self.riders, self.all_orders)
        
        # Thought
        thought = self.platform.think(self.riders, env_state)
        
        # Actions
        actions = []
        
        # 计算收益
        daily_orders = [o for o in self.all_orders if o.delivered and env_state["day"] == env_state["day"]]
        profit = self.platform.calc_profit(daily_orders)
        actions.append(f"日收益: {profit:.1f}元")
        
        # 考虑解雇
        for rider in self.riders:
            if self.platform.consider_fire_rider(rider):
                actions.append(f"解雇: {rider.agent_id}")
        
        # 缴税
        if env_state["day"] % 7 == 0:
            tax = self.platform.pay_tax(self.government)
            actions.append(f"缴税: {tax:.1f}元")
        
        action_desc = "; ".join(actions)
        print(f"  平台: {action_desc}")
        
        self.logger.log_agent_action(
            env_state["day"], env_state["hour"], "Platform", self.platform.agent_id,
            obs, thought, action_desc
        )
    
    def _government_workflow(self, env_state: Dict):
        """政府工作流"""
        # Observation
        obs = self.government.observe(env_state, self.riders)
        
        # Thought
        thought = self.government.think(env_state, self.riders)
        
        # Actions
        actions = []
        
        # 高温补贴
        subsidy = self.government.decide_subsidy(env_state, self.riders)
        if subsidy > 0:
            actions.append(f"发放补贴: {subsidy:.1f}元")
            print(f"  政府发放高温补贴: {subsidy:.1f}元")
        
        # 增设纳凉点
        if self.government.decide_build_shelter(env_state, self.riders):
            self.environment.add_shelter(0.1)
            actions.append("增设纳凉点")
            print(f"  政府增设纳凉点，覆盖率提升至{self.environment.shelter_rate:.2f}")
        
        action_desc = "; ".join(actions) if actions else "保持观察"
        
        self.logger.log_agent_action(
            env_state["day"], env_state["hour"], "Government", self.government.agent_id,
            obs, thought, action_desc
        )
    
    def _generate_final_report(self):
        """生成最终报告"""
        print("\n=== 最终报告 ===")
        
        # 骑手状态
        print("骑手最终状态:")
        for rider in self.riders:
            status = rider.get_status()
            print(f"  {rider.agent_id}: 健康{status['health']:.1f}/10, "
                  f"幸福感{status['happiness']:.1f}/10, "
                  f"资产{status['money']:.1f}元, "
                  f"完成订单{status['orders_completed']}个")
        
        # 整体统计
        total_orders = len(self.all_orders)
        completed_orders = len([o for o in self.all_orders if o.delivered])
        avg_health = np.mean([r.health for r in self.riders])
        avg_happiness = np.mean([r.happiness for r in self.riders])
        total_complaints = sum(len(r.complaints) for r in self.riders)
        
        print(f"\n整体统计:")
        print(f"  总订单数: {total_orders}")
        print(f"  完成订单: {completed_orders}")
        print(f"  完成率: {completed_orders/max(1,total_orders):.1%}")
        print(f"  平均健康: {avg_health:.1f}/10")
        print(f"  平均幸福感: {avg_happiness:.1f}/10")
        print(f"  总投诉: {total_complaints}")
        print(f"  政府补贴: {self.government.subsidies_paid:.1f}元")
        print(f"  纳凉点覆盖率: {self.environment.shelter_rate:.2f}")

def main():
    """测试主函数"""
    print("简化版仿真测试")
    
    # 运行小规模测试
    sim = SimpleHeatWeatherSimulation(num_customers=3, num_riders=2, simulation_days=2)
    sim.run_simple_simulation()

if __name__ == "__main__":
    main()
