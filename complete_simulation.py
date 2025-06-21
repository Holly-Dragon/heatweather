"""
完整版多主体仿真系统（不使用LangGraph）
基于文档需求实现的极端高温下外卖配送仿真
"""

import random
import numpy as np
import json
from typing import Dict, List
from environment import Environment
from agents import Customer, Rider, Platform, Government, Order
from utils import SimulationLogger

class CompleteHeatWeatherSimulation:
    """完整版极端高温仿真系统"""
    
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
        
        print(f"仿真系统初始化完成:")
        print(f"- 客户数量: {num_customers}")
        print(f"- 骑手数量: {num_riders}")
        print(f"- 仿真天数: {simulation_days}")
        print(f"- 预计总时长: {simulation_days * 24}小时")
        
    def run_complete_simulation(self):
        """运行完整仿真"""
        print("\n" + "="*60)
        print("开始完整版极端高温多主体仿真")
        print("="*60)
        
        try:
            for day in range(self.simulation_days):
                print(f"\n📅 === Day {day + 1}/{self.simulation_days} ===")
                self._simulate_one_day(day)
                
                # 每日报告
                if day % 7 == 0 or day == self.simulation_days - 1:  # 每周或最后一天报告
                    self._print_progress_report(day)
            
            # 最终统计和分析
            self._generate_final_analysis()
            
        except KeyboardInterrupt:
            print("\n⚠️ 仿真被用户中断")
            self._generate_final_analysis()
        except Exception as e:
            print(f"\n❌ 仿真过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _simulate_one_day(self, day: int):
        """模拟一天的活动"""
        daily_orders = []
        daily_actions = []
        
        for hour in range(24):
            self.environment.current_day = day
            self.environment.current_hour = hour
            env_state = self.environment.get_environment_state()
            
            # 温度警报
            temp = env_state['temperature']
            if temp > 42:
                print(f"🔥 {hour:02d}:00 极端高温警报！温度: {temp:.1f}°C")
            elif temp > 38:
                print(f"🌡️ {hour:02d}:00 高温预警，温度: {temp:.1f}°C")
            
            # 1. 客户工作流 - 用餐时间点外卖
            if env_state["is_meal_time"]:
                customer_orders = self._execute_customer_workflow(env_state)
                daily_orders.extend(customer_orders)
                daily_actions.extend([f"客户下单{len(customer_orders)}个"])
            
            # 2. 骑手工作流 - 每小时都可能有行动
            rider_actions = self._execute_rider_workflow(env_state)
            daily_actions.extend(rider_actions)
            
            # 3. 企业工作流 - 每天结束时
            if hour == 23:
                platform_actions = self._execute_platform_workflow(env_state, day)
                daily_actions.extend(platform_actions)
            
            # 4. 政府工作流 - 每天晚上22点
            if hour == 22:
                government_actions = self._execute_government_workflow(env_state, day)
                daily_actions.extend(government_actions)
        
        # 记录日统计
        final_env_state = self.environment.get_environment_state()
        self.logger.log_daily_stats(
            day, self.riders, self.customers, self.all_orders,
            final_env_state, self.government, self.platform
        )
    
    def _execute_customer_workflow(self, env_state: Dict) -> List[Order]:
        """执行客户工作流"""
        new_orders = []
        
        for customer in self.customers:
            # Observation - Thought - Action 流程
            obs = customer.observe(env_state)
            thought = customer.think(env_state)
            
            # 决定是否下单
            order = customer.decide_order(env_state)
            if order:
                new_orders.append(order)
                self.current_orders.append(order)
                self.all_orders.append(order)
                
                print(f"  📱 客户{customer.agent_id[-1]}下单: {order.cost:.0f}元, {order.distance:.1f}km")
            
            # 记录Agent行为
            action = f"下单ID:{order.order_id[:8]}" if order else "未下单"
            self.logger.log_agent_action(
                env_state["day"], env_state["hour"], "Customer", customer.agent_id,
                obs, thought, action
            )
        
        return new_orders
    
    def _execute_rider_workflow(self, env_state: Dict) -> List[str]:
        """执行骑手工作流"""
        actions = []
        available_orders = [o for o in self.current_orders if not o.rider_id]
        
        for rider in self.riders:
            if not rider.on_duty:
                continue
            
            # Observation - Thought - Action 流程
            obs = rider.observe(env_state, available_orders)
            thought = rider.think(env_state, available_orders)
            action_type = rider.decide_action(env_state, available_orders)
            
            action_desc = ""
            
            if action_type == "deliver" and available_orders:
                # 选择一个订单配送
                order = random.choice(available_orders)
                result = rider.deliver_order(order, env_state)
                available_orders.remove(order)
                if order in self.current_orders:
                    self.current_orders.remove(order)
                
                # 客户评分和小费
                customer = next(c for c in self.customers if c.agent_id == order.customer_id)
                rating = customer.rate_order(order, rider.health)
                tip = customer.decide_tip(order, rider.health)
                
                action_desc = f"配送完成，收入{result['income']:.0f}元，评分{rating}星"
                actions.append(f"🚴 骑手{rider.agent_id[-1]}: {action_desc}")
                
                # 健康警报
                if rider.health < 3:
                    print(f"  ⚠️ 骑手{rider.agent_id[-1]}健康状况严重: {rider.health:.1f}/10")
                elif rider.health < 5:
                    print(f"  ⚡ 骑手{rider.agent_id[-1]}健康状况较差: {rider.health:.1f}/10")
                    
            elif action_type == "rest":
                rider.rest(env_state)
                action_desc = "休息恢复"
                
            elif action_type == "complain":
                complaint = rider.complain(env_state)
                action_desc = "投诉工作条件"
                actions.append(f"📢 骑手{rider.agent_id[-1]}投诉: 健康{rider.health:.1f}, 温度{env_state['temperature']:.1f}°C")
                
            # 记录Agent行为
            self.logger.log_agent_action(
                env_state["day"], env_state["hour"], "Rider", rider.agent_id,
                obs, thought, action_desc
            )
        
        return actions
    
    def _execute_platform_workflow(self, env_state: Dict, day: int) -> List[str]:
        """执行平台工作流"""
        actions = []
        
        # Observation - Thought - Action 流程
        obs = self.platform.observe(self.riders, self.all_orders)
        thought = self.platform.think(self.riders, env_state)
        
        # 计算日收益
        daily_orders = [o for o in self.all_orders if o.delivered and hasattr(o, 'day') and getattr(o, 'day', day) == day]
        profit = self.platform.calc_profit(self.all_orders)  # 简化处理
        actions.append(f"💰 平台日收益: {profit:.0f}元")
        
        # 考虑解雇表现差的骑手
        fired_count = 0
        for rider in self.riders:
            if self.platform.consider_fire_rider(rider):
                fired_count += 1
        
        if fired_count > 0:
            actions.append(f"🚫 解雇{fired_count}名骑手")
        
        # 每周缴税
        if day % 7 == 0 and day > 0:
            tax = self.platform.pay_tax(self.government)
            actions.append(f"💸 向政府缴税: {tax:.0f}元")
        
        # 记录平台行为
        action_desc = "; ".join(actions) if actions else "日常运营"
        self.logger.log_agent_action(
            env_state["day"], env_state["hour"], "Platform", self.platform.agent_id,
            obs, thought, action_desc
        )
        
        return actions
    
    def _execute_government_workflow(self, env_state: Dict, day: int) -> List[str]:
        """执行政府工作流"""
        actions = []
        
        # Observation - Thought - Action 流程
        obs = self.government.observe(env_state, self.riders)
        thought = self.government.think(env_state, self.riders)
        
        # 高温补贴决策
        subsidy = self.government.decide_subsidy(env_state, self.riders)
        if subsidy > 0:
            actions.append(f"🎁 政府发放高温补贴: {subsidy:.0f}元")
        
        # 增设纳凉点决策
        if self.government.decide_build_shelter(env_state, self.riders):
            self.environment.add_shelter(0.1)  # 第二天生效
            actions.append(f"🏠 政府增设纳凉点，覆盖率提升至{self.environment.shelter_rate:.2f}")
        
        # 记录政府行为
        action_desc = "; ".join(actions) if actions else "保持观察"
        self.logger.log_agent_action(
            env_state["day"], env_state["hour"], "Government", self.government.agent_id,
            obs, thought, action_desc
        )
        
        return actions
    
    def _print_progress_report(self, day: int):
        """打印进度报告"""
        print(f"\n📊 === Day {day + 1} 进度报告 ===")
        
        # 骑手状态
        print("🏥 骑手健康状况:")
        for i, rider in enumerate(self.riders):
            status_icon = "🤒" if rider.health < 3 else "😰" if rider.health < 5 else "😊" if rider.health < 8 else "💪"
            duty_status = "在职" if rider.on_duty else "离职"
            print(f"  {status_icon} 骑手{i}: 健康{rider.health:.1f}/10, 幸福感{rider.happiness:.1f}/10, 资产{rider.money:.0f}元 ({duty_status})")
        
        # 订单统计
        completed_orders = len([o for o in self.all_orders if o.delivered])
        completion_rate = completed_orders / max(1, len(self.all_orders))
        print(f"📦 订单: 总数{len(self.all_orders)}, 完成{completed_orders}, 完成率{completion_rate:.1%}")
        
        # 政府措施
        total_complaints = sum(len(r.complaints) for r in self.riders)
        print(f"🏛️ 政府: 补贴{self.government.subsidies_paid:.0f}元, 纳凉点覆盖率{self.environment.shelter_rate:.2f}, 收到投诉{total_complaints}次")
    
    def _generate_final_analysis(self):
        """生成最终分析报告"""
        print("\n" + "="*60)
        print("📈 最终分析报告")
        print("="*60)
        
        # 保存详细日志
        self.logger.save_logs(f"complete_simulation_logs_{self.simulation_days}days.json")
        
        # 基础统计
        total_orders = len(self.all_orders)
        completed_orders = len([o for o in self.all_orders if o.delivered])
        completion_rate = completed_orders / max(1, total_orders)
        
        avg_health = np.mean([r.health for r in self.riders])
        avg_happiness = np.mean([r.happiness for r in self.riders])
        total_complaints = sum(len(r.complaints) for r in self.riders)
        
        active_riders = len([r for r in self.riders if r.on_duty])
        
        print(f"📊 基础统计:")
        print(f"  - 仿真天数: {self.simulation_days}")
        print(f"  - 总订单数: {total_orders}")
        print(f"  - 完成订单: {completed_orders}")
        print(f"  - 订单完成率: {completion_rate:.1%}")
        print(f"  - 在职骑手: {active_riders}/{self.num_riders}")
        
        print(f"\n🏥 骑手健康与幸福感:")
        print(f"  - 平均健康水平: {avg_health:.1f}/10")
        print(f"  - 平均幸福感: {avg_happiness:.1f}/10")
        print(f"  - 总投诉数: {total_complaints}")
        
        print(f"\n🏛️ 政府措施效果:")
        print(f"  - 总补贴支出: {self.government.subsidies_paid:.0f}元")
        print(f"  - 新建纳凉点: {self.government.shelters_built}个")
        print(f"  - 最终纳凉点覆盖率: {self.environment.shelter_rate:.2f}")
        print(f"  - 政府预算余额: {self.government.budget:.0f}元")
        
        print(f"\n💰 经济效果:")
        print(f"  - 平台总资金: {self.platform.cash:.0f}元")
        
        # 评估结果
        print(f"\n🎯 政策效果评估:")
        
        # 健康评估
        if avg_health >= 8:
            health_grade = "优秀 ✅"
        elif avg_health >= 6:
            health_grade = "良好 👍"
        elif avg_health >= 4:
            health_grade = "一般 ⚠️"
        else:
            health_grade = "较差 ❌"
        print(f"  - 骑手健康水平: {health_grade}")
        
        # 幸福感评估
        if avg_happiness >= 7:
            happiness_grade = "优秀 😊"
        elif avg_happiness >= 5:
            happiness_grade = "良好 🙂"
        elif avg_happiness >= 3:
            happiness_grade = "一般 😐"
        else:
            happiness_grade = "较差 😟"
        print(f"  - 骑手幸福感: {happiness_grade}")
        
        # 服务质量评估
        if completion_rate >= 0.9:
            service_grade = "优秀 🎉"
        elif completion_rate >= 0.8:
            service_grade = "良好 👍"
        elif completion_rate >= 0.7:
            service_grade = "一般 ⚠️"
        else:
            service_grade = "较差 ❌"
        print(f"  - 服务完成率: {service_grade}")
        
        # 政策建议
        print(f"\n💡 政策建议:")
        suggestions = []
        
        if avg_health < 5:
            suggestions.append("📌 骑手健康状况堪忧，建议增加高温补贴和强制休息时间")
        if total_complaints > self.simulation_days * 2:
            suggestions.append("📌 投诉数量过多，建议改善工作环境和条件")
        if completion_rate < 0.8:
            suggestions.append("📌 服务完成率偏低，建议平衡工作强度与健康保护")
        if self.environment.shelter_rate < 0.5:
            suggestions.append("📌 纳凉点覆盖率仍需提升，建议继续增设基础设施")
        if active_riders < self.num_riders * 0.8:
            suggestions.append("📌 骑手流失率较高，建议改善工作待遇和保障")
        
        if not suggestions:
            suggestions.append("📌 各项指标表现良好，建议保持现有政策")
        
        for suggestion in suggestions:
            print(f"  {suggestion}")
        
        # 保存分析结果
        analysis_result = {
            "simulation_days": self.simulation_days,
            "total_orders": total_orders,
            "completion_rate": completion_rate,
            "avg_health": avg_health,
            "avg_happiness": avg_happiness,
            "total_complaints": total_complaints,
            "government_subsidies": self.government.subsidies_paid,
            "shelters_built": self.government.shelters_built,
            "final_shelter_rate": self.environment.shelter_rate,
            "active_riders": active_riders,
            "suggestions": suggestions
        }
        
        with open(f"analysis_result_{self.simulation_days}days.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细日志已保存至: complete_simulation_logs_{self.simulation_days}days.json")
        print(f"📋 分析结果已保存至: analysis_result_{self.simulation_days}days.json")
        
        # 尝试生成图表
        try:
            self.logger.plot_simulation_results()
            self.logger.plot_rider_analysis()
            print("📊 图表已生成: simulation_results.png, rider_analysis.png")
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")

def main():
    """主函数"""
    print("🌡️ 极端高温下外卖配送多主体仿真系统")
    print("基于Agent的政府政策效果评估")
    
    # 获取仿真参数
    try:
        customers = int(input("请输入客户数量 (默认10): ") or "10")
        riders = int(input("请输入骑手数量 (默认3): ") or "3")
        days = int(input("请输入仿真天数 (默认30): ") or "30")
    except ValueError:
        customers, riders, days = 10, 3, 30
        print("使用默认参数: 10个客户, 3个骑手, 30天")
    
    # 创建并运行仿真
    simulation = CompleteHeatWeatherSimulation(customers, riders, days)
    simulation.run_complete_simulation()

if __name__ == "__main__":
    main()
