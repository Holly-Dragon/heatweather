"""
工具函数和数据记录
"""

import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class SimulationLogger:
    """仿真数据记录器"""
    
    def __init__(self):
        self.logs = []
        self.daily_stats = []
        self.rider_stats = []
        self.customer_stats = []
        self.environment_stats = []
        
    def log_agent_action(self, day: int, hour: int, agent_type: str, agent_id: str, 
                        observation: str, thought: str, action: str):
        """记录Agent的行为"""
        log_entry = {
            "timestamp": f"Day{day}-Hour{hour}",
            "day": day,
            "hour": hour,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "observation": observation,
            "thought": thought,
            "action": action
        }
        self.logs.append(log_entry)
        
    def log_daily_stats(self, day: int, riders: List, customers: List, 
                       orders: List, environment_state: Dict, government, platform):
        """记录每日统计数据"""
        
        # 骑手统计
        rider_data = []
        for rider in riders:
            rider_data.append({
                "day": day,
                "rider_id": rider.agent_id,
                "health": rider.health,
                "money": rider.money,
                "happiness": rider.happiness,
                "orders_completed": rider.orders_completed,
                "daily_income": rider.daily_income,
                "on_duty": rider.on_duty,
                "complaints": len(rider.complaints)
            })
        self.rider_stats.extend(rider_data)
        
        # 整体日统计
        daily_stat = {
            "day": day,
            "avg_temperature": np.mean([environment_state["temperature"]]),  # 简化处理
            "total_orders": len(orders),
            "completed_orders": len([o for o in orders if o.delivered]),
            "avg_rider_health": np.mean([r.health for r in riders]),
            "avg_rider_happiness": np.mean([r.happiness for r in riders]),
            "total_complaints": sum(len(r.complaints) for r in riders),
            "government_subsidies": government.subsidies_paid,
            "shelters_built": government.shelters_built,
            "platform_revenue": platform.daily_revenue,
            "shelter_rate": environment_state["shelter_rate"]
        }
        self.daily_stats.append(daily_stat)
        
    def save_logs(self, filename: str = "simulation_logs.json"):
        """保存日志到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "logs": self.logs,
                "daily_stats": self.daily_stats,
                "rider_stats": self.rider_stats
            }, f, ensure_ascii=False, indent=2)
            
    def print_daily_summary(self, day: int):
        """打印每日摘要"""
        print(f"\n=== Day {day} 仿真摘要 ===")
        
        if self.daily_stats:
            stats = self.daily_stats[-1]
            print(f"平均温度: {stats['avg_temperature']:.1f}°C")
            print(f"订单总数: {stats['total_orders']}, 完成: {stats['completed_orders']}")
            print(f"骑手平均健康: {stats['avg_rider_health']:.1f}/10")
            print(f"骑手平均幸福感: {stats['avg_rider_happiness']:.1f}/10")
            print(f"投诉总数: {stats['total_complaints']}")
            print(f"政府补贴: {stats['government_subsidies']:.1f}元")
            print(f"新建纳凉点: {stats['shelters_built']}个")
            print(f"阴凉覆盖率: {stats['shelter_rate']:.2f}")
        
        # 打印最近的Agent行为
        recent_logs = [log for log in self.logs if log['day'] == day]
        print(f"\n今日Agent行为记录 ({len(recent_logs)}条):")
        for log in recent_logs[-10:]:  # 只显示最近10条
            print(f"  {log['timestamp']} [{log['agent_type']}-{log['agent_id']}]")
            print(f"    观察: {log['observation']}")
            print(f"    思考: {log['thought']}")
            print(f"    行动: {log['action']}")
            print()
    
    def plot_simulation_results(self):
        """绘制仿真结果图表"""
        if not self.daily_stats:
            print("没有数据可绘制")
            return
            
        df = pd.DataFrame(self.daily_stats)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('极端高温下外卖配送仿真结果', fontsize=16)
        
        # 温度和健康趋势
        axes[0, 0].plot(df['day'], df['avg_temperature'], 'r-', label='平均温度')
        axes[0, 0].set_ylabel('温度 (°C)')
        axes[0, 0].set_title('平均温度变化')
        axes[0, 0].grid(True, alpha=0.3)
        
        ax1_twin = axes[0, 0].twinx()
        ax1_twin.plot(df['day'], df['avg_rider_health'], 'b-', label='骑手平均健康')
        ax1_twin.set_ylabel('健康水平')
        ax1_twin.legend(loc='upper right')
        axes[0, 0].legend(loc='upper left')
        
        # 幸福感变化
        axes[0, 1].plot(df['day'], df['avg_rider_happiness'], 'g-', marker='o')
        axes[0, 1].set_ylabel('幸福感')
        axes[0, 1].set_title('骑手平均幸福感变化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 订单和投诉
        axes[0, 2].bar(df['day'], df['total_orders'], alpha=0.7, label='总订单')
        axes[0, 2].bar(df['day'], df['completed_orders'], alpha=0.7, label='完成订单')
        axes[0, 2].set_ylabel('订单数量')
        axes[0, 2].set_title('订单完成情况')
        axes[0, 2].legend()
        
        # 政府措施效果
        axes[1, 0].bar(df['day'], df['government_subsidies'], alpha=0.7, color='orange')
        axes[1, 0].set_ylabel('补贴金额 (元)')
        axes[1, 0].set_title('政府高温补贴')
        
        # 基础设施建设
        axes[1, 1].plot(df['day'], df['shelter_rate'], 'purple', marker='s')
        axes[1, 1].set_ylabel('覆盖率')
        axes[1, 1].set_title('阴凉点覆盖率变化')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 投诉情况
        axes[1, 2].bar(df['day'], df['total_complaints'], alpha=0.7, color='red')
        axes[1, 2].set_ylabel('投诉数量')
        axes[1, 2].set_title('每日投诉统计')
        
        for ax in axes.flat:
            ax.set_xlabel('仿真天数')
            
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_rider_analysis(self):
        """分析单个骑手的表现"""
        if not self.rider_stats:
            print("没有骑手数据可分析")
            return
            
        df = pd.DataFrame(self.rider_stats)
        riders = df['rider_id'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('骑手个体表现分析', fontsize=16)
        
        # 健康变化
        for rider_id in riders:
            rider_data = df[df['rider_id'] == rider_id]
            axes[0, 0].plot(rider_data['day'], rider_data['health'], marker='o', label=f'骑手{rider_id}')
        axes[0, 0].set_ylabel('健康水平')
        axes[0, 0].set_title('骑手健康变化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 幸福感变化
        for rider_id in riders:
            rider_data = df[df['rider_id'] == rider_id]
            axes[0, 1].plot(rider_data['day'], rider_data['happiness'], marker='o', label=f'骑手{rider_id}')
        axes[0, 1].set_ylabel('幸福感')
        axes[0, 1].set_title('骑手幸福感变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收入变化
        for rider_id in riders:
            rider_data = df[df['rider_id'] == rider_id]
            axes[1, 0].plot(rider_data['day'], rider_data['money'], marker='o', label=f'骑手{rider_id}')
        axes[1, 0].set_ylabel('总资产 (元)')
        axes[1, 0].set_title('骑手资产变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 订单完成情况
        for rider_id in riders:
            rider_data = df[df['rider_id'] == rider_id]
            axes[1, 1].plot(rider_data['day'], rider_data['orders_completed'], marker='o', label=f'骑手{rider_id}')
        axes[1, 1].set_ylabel('累计订单数')
        axes[1, 1].set_title('骑手订单完成数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('仿真天数')
            
        plt.tight_layout()
        plt.savefig('rider_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_agent_workflow(agent, step_name: str):
    """打印Agent工作流程"""
    print(f"\n--- {agent.agent_type} {agent.agent_id} - {step_name} ---")
    if hasattr(agent, 'observations') and agent.observations:
        print(f"最新观察: {agent.observations[-1]}")
    if hasattr(agent, 'thoughts') and agent.thoughts:
        print(f"最新思考: {agent.thoughts[-1]}")
    if hasattr(agent, 'actions') and agent.actions:
        print(f"最新行动: {agent.actions[-1]}")
