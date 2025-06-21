"""
环境规则模拟器
模拟极端高温天气环境和各种参数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta

class Environment:
    """环境规则模拟器"""
    
    def __init__(self):
        self.current_day = 0
        self.current_hour = 6  # 从早上6点开始
        self.shelter_rate = 0.15  # 初始阴凉覆盖率
        self.rest_rate = 0.6      # 休息区覆盖率
        self.tax_rate = 0.1       # 税率
        self.temperature_curve = self._generate_temperature_curve()
        
    def _generate_temperature_curve(self) -> List[float]:
        """生成24小时温度曲线（极端高温场景）"""
        hours = np.arange(24)
        # 基础温度曲线：夜间35度，白天最高45度
        base_temp = 35 + 10 * np.sin((hours - 6) * np.pi / 12)
        # 添加随机波动
        noise = np.random.normal(0, 2, 24)
        temperatures = base_temp + noise
        # 确保最低温度不低于32度，最高不超过48度
        temperatures = np.clip(temperatures, 32, 48)
        return temperatures.tolist()
    
    def get_current_temperature(self) -> float:
        """获取当前温度"""
        return self.temperature_curve[self.current_hour]
    
    def get_order_distance(self) -> float:
        """生成订单距离，遵循Zipf分布"""
        # 使用Zipf分布生成1-10km的距离
        zipf_sample = np.random.zipf(1.5)
        distance = min(zipf_sample, 10)  # 限制最大距离为10km
        return max(1, distance)  # 最小距离为1km
    
    def get_order_cost(self) -> float:
        """生成订单金额"""
        return random.uniform(15, 50)
    
    def is_meal_time(self) -> bool:
        """判断是否为用餐时间"""
        meal_times = [
            (7, 9),   # 早餐
            (11, 13), # 午餐
            (17, 19)  # 晚餐
        ]
        return any(start <= self.current_hour < end for start, end in meal_times)
    
    def advance_hour(self):
        """推进一小时"""
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
            # 每天重新生成温度曲线
            self.temperature_curve = self._generate_temperature_curve()
    
    def add_shelter(self, increase_rate: float):
        """政府增设纳凉点"""
        self.shelter_rate = min(1.0, self.shelter_rate + increase_rate)
        print(f"Day {self.current_day}: 政府增设纳凉点，阴凉覆盖率提升至 {self.shelter_rate:.2f}")
    
    def get_environment_state(self) -> Dict:
        """获取环境状态"""
        return {
            "day": self.current_day,
            "hour": self.current_hour,
            "temperature": self.get_current_temperature(),
            "shelter_rate": self.shelter_rate,
            "rest_rate": self.rest_rate,
            "is_meal_time": self.is_meal_time()
        }
    
    def plot_temperature_curve(self):
        """绘制当天温度曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(range(24), self.temperature_curve, 'r-', linewidth=2, label='温度曲线')
        plt.axhline(y=35, color='orange', linestyle='--', alpha=0.7, label='高温预警线')
        plt.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='极端高温线')
        plt.xlabel('小时')
        plt.ylabel('温度 (°C)')
        plt.title(f'Day {self.current_day} 温度变化曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.show()
