"""
所有Agent的定义
包括：客户、外卖小哥、平台、政府
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import uuid
from collections import deque
import numpy as np

class AgentState:
    """Agent基础状态类"""
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.observations = []
        self.thoughts = []
        self.actions = []
        
    def add_observation(self, obs: str):
        self.observations.append(f"[{self.agent_type}-{self.agent_id}] 观察: {obs}")
        
    def add_thought(self, thought: str):
        self.thoughts.append(f"[{self.agent_type}-{self.agent_id}] 思考: {thought}")
        
    def add_action(self, action: str):
        self.actions.append(f"[{self.agent_type}-{self.agent_id}] 行动: {action}")

@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    customer_id: str
    time: int
    cost: float
    distance: float
    rider_id: Optional[str] = None
    delivered: bool = False
    rating: Optional[int] = None
    tip: float = 0.0

class Customer(AgentState):
    """点外卖群体Agent"""
    
    def __init__(self, customer_id: str):
        super().__init__(customer_id, "Customer")
        self.last_ratings = deque(maxlen=5)  # 最近5次评分
        self.order_history = []
        
    def observe(self, environment_state: Dict) -> str:
        """观察环境状态"""
        temp = environment_state["temperature"]
        avg_rating = np.mean(self.last_ratings) if self.last_ratings else 5.0
        
        obs = f"当前温度{temp:.1f}°C，我的前几次外卖评分均值为{avg_rating:.1f}"
        self.add_observation(obs)
        return obs
    
    def think(self, environment_state: Dict) -> str:
        """思考决策"""
        temp = environment_state["temperature"]
        avg_rating = np.mean(self.last_ratings) if self.last_ratings else 5.0
        is_meal_time = environment_state["is_meal_time"]
        
        # 高温会降低点外卖概率，评分低也会降低概率
        temp_factor = max(0.1, 1.0 - (temp - 30) / 20)  # 温度越高概率越低
        rating_factor = avg_rating / 5.0  # 评分越高概率越高
        meal_factor = 1.5 if is_meal_time else 0.3  # 用餐时间概率更高
        
        order_prob = temp_factor * rating_factor * meal_factor
        
        thought = f"考虑到温度{temp:.1f}°C，服务评分{avg_rating:.1f}，用餐时间{is_meal_time}，点外卖概率为{order_prob:.2f}"
        self.add_thought(thought)
        return thought
    
    def decide_order(self, environment_state: Dict) -> Optional[Order]:
        """决定是否下单"""
        temp = environment_state["temperature"]
        avg_rating = np.mean(self.last_ratings) if self.last_ratings else 5.0
        is_meal_time = environment_state["is_meal_time"]
        
        # 计算下单概率
        temp_factor = max(0.1, 1.0 - (temp - 30) / 20)
        rating_factor = avg_rating / 5.0
        meal_factor = 1.5 if is_meal_time else 0.3
        order_prob = temp_factor * rating_factor * meal_factor * 0.3  # 基础概率
        
        if random.random() < order_prob:
            from environment import Environment
            env = Environment()
            order = Order(
                order_id=str(uuid.uuid4()),
                customer_id=self.agent_id,
                time=environment_state["hour"],
                cost=env.get_order_cost(),
                distance=env.get_order_distance()
            )
            self.order_history.append(order)
            action = f"下单 - 订单ID:{order.order_id[:8]}, 金额:{order.cost:.1f}元, 距离:{order.distance:.1f}km"
            self.add_action(action)
            return order
        return None
    
    def rate_order(self, order: Order, rider_health: float) -> int:
        """对订单评分"""
        # 基础评分
        base_rating = 5
        
        # 根据骑手健康状况调整评分
        if rider_health < 5:
            base_rating -= 1  # 健康状况不佳影响服务质量
        if rider_health < 3:
            base_rating -= 1  # 健康状况很差
            
        # 添加随机因素
        rating = max(1, min(5, base_rating + random.randint(-1, 1)))
        
        self.last_ratings.append(rating)
        order.rating = rating
        
        action = f"给订单{order.order_id[:8]}评分{rating}星"
        self.add_action(action)
        return rating
    
    def decide_tip(self, order: Order, rider_health: float) -> float:
        """决定是否给小费"""
        if rider_health < 3:  # 骑手健康状况很差，给小费表示同情
            tip = random.uniform(2, 5)
        elif order.rating and order.rating >= 4:  # 评分高给小费
            tip = random.uniform(1, 3)
        else:
            tip = 0
            
        order.tip = tip
        if tip > 0:
            action = f"给订单{order.order_id[:8]}小费{tip:.1f}元"
            self.add_action(action)
        return tip

class Rider(AgentState):
    """外卖小哥Agent"""
    
    def __init__(self, rider_id: str):
        super().__init__(rider_id, "Rider")
        self.health = 10.0  # 健康状况
        self.money = 1000.0  # 现有金额
        self.on_duty = True  # 是否在职
        self.happiness = 5.0  # 幸福感
        self.orders_completed = 0
        self.complaints = []
        self.daily_income = 0.0
        
    def observe(self, environment_state: Dict, available_orders: List[Order]) -> str:
        """观察环境状态"""
        temp = environment_state["temperature"]
        shelter_rate = environment_state["shelter_rate"]
        order_count = len(available_orders)
        
        obs = f"当前温度{temp:.1f}°C，阴凉覆盖率{shelter_rate:.1f}，可接订单{order_count}个，我的健康状况{self.health:.1f}/10"
        self.add_observation(obs)
        return obs
    
    def think(self, environment_state: Dict, available_orders: List[Order]) -> str:
        """思考决策"""
        temp = environment_state["temperature"]
        
        # 健康状况判断
        if self.health < 2:
            thought = f"健康状况严重恶化({self.health:.1f}/10)，必须休息或就医"
        elif self.health < 5:
            thought = f"健康状况较差({self.health:.1f}/10)，需要谨慎工作"
        elif temp > 40:
            thought = f"极端高温{temp:.1f}°C，工作风险很高"
        elif len(available_orders) > 0:
            thought = f"有{len(available_orders)}个订单可接，考虑接单赚钱"
        else:
            thought = "当前没有订单，等待或休息"
            
        self.add_thought(thought)
        return thought
    
    def decide_action(self, environment_state: Dict, available_orders: List[Order]) -> str:
        """决定行动"""
        if not self.on_duty:
            return "off_duty"
            
        temp = environment_state["temperature"]
        
        # 健康状况太差，必须休息
        if self.health < 2:
            return "rest"
            
        # 极端高温且健康状况不好，倾向于休息
        if temp > 42 and self.health < 6:
            if random.random() < 0.7:
                return "rest"
                
        # 有订单且健康状况允许，考虑接单
        if available_orders and self.health >= 3:
            # 根据健康状况和温度决定接单概率
            health_factor = self.health / 10.0
            temp_factor = max(0.1, 1.0 - (temp - 35) / 15)
            work_prob = health_factor * temp_factor
            
            if random.random() < work_prob:
                return "deliver"
                
        # 健康状况很差，考虑投诉
        if self.health < 3 and random.random() < 0.3:
            return "complain"
            
        return "rest"
    
    def deliver_order(self, order: Order, environment_state: Dict) -> Dict:
        """接单送餐"""
        temp = environment_state["temperature"]
        shelter_rate = environment_state["shelter_rate"]
        
        # 计算健康损失
        K = 0.2  # 健康损失系数
        health_loss = (temp - 35) * K * order.distance * (1 - shelter_rate)
        health_loss = max(0, health_loss)  # 确保不为负数
        
        self.health -= health_loss
        self.health = max(0, self.health)  # 确保健康值不为负
        
        # 计算收入（基础收入 + 小费）
        base_income = order.cost * 0.2  # 假设骑手收入为订单金额的20%
        total_income = base_income + order.tip
        self.money += total_income
        self.daily_income += total_income
        self.orders_completed += 1
        
        order.rider_id = self.agent_id
        order.delivered = True
        
        # 更新幸福感
        self.update_happiness(environment_state)
        
        action = f"完成订单{order.order_id[:8]}，收入{total_income:.1f}元，健康损失{health_loss:.1f}，当前健康{self.health:.1f}/10"
        self.add_action(action)
        
        return {
            "order": order,
            "income": total_income,
            "health_loss": health_loss
        }
    
    def rest(self, environment_state: Dict):
        """休息恢复"""
        rest_rate = environment_state["rest_rate"]
        temp = environment_state["temperature"]
        
        # 休息恢复健康
        if temp > 40:
            recovery = 0.5 * rest_rate  # 高温下恢复较慢
        else:
            recovery = 1.0 * rest_rate
            
        self.health += recovery
        self.health = min(10, self.health)  # 确保健康值不超过10
        
        action = f"休息恢复健康{recovery:.1f}，当前健康{self.health:.1f}/10"
        self.add_action(action)
    
    def complain(self, environment_state: Dict):
        """投诉"""
        complaint = {
            "rider_id": self.agent_id,
            "day": environment_state["day"],
            "hour": environment_state["hour"],
            "health": self.health,
            "temperature": environment_state["temperature"],
            "reason": "极端高温工作条件恶劣，健康受损严重"
        }
        self.complaints.append(complaint)
        
        action = f"投诉工作条件恶劣，当前健康{self.health:.1f}/10，温度{environment_state['temperature']:.1f}°C"
        self.add_action(action)
        return complaint
    
    def update_happiness(self, environment_state: Dict):
        """更新幸福感"""
        # 基础幸福感受多个因素影响
        health_factor = self.health / 10.0  # 健康因素
        money_factor = min(1.0, self.money / 2000.0)  # 收入因素
        temp_factor = max(0.1, 1.0 - (environment_state["temperature"] - 30) / 20)  # 温度因素
        
        self.happiness = (health_factor * 0.4 + money_factor * 0.3 + temp_factor * 0.3) * 10
        self.happiness = max(0, min(10, self.happiness))
    
    def get_status(self) -> Dict:
        """获取状态信息"""
        return {
            "rider_id": self.agent_id,
            "health": self.health,
            "money": self.money,
            "happiness": self.happiness,
            "on_duty": self.on_duty,
            "orders_completed": self.orders_completed,
            "daily_income": self.daily_income
        }

class Platform(AgentState):
    """外卖平台Agent"""
    
    def __init__(self):
        super().__init__("platform", "Platform")
        self.cash = 5000.0  # 初始资金
        self.tax_time = 0  # 缴税时间记录
        self.daily_revenue = 0.0
        self.rider_pay_rate = 0.2  # 骑手分成比例
        
    def observe(self, riders: List[Rider], orders: List[Order]) -> str:
        """观察骑手表现和订单情况"""
        active_riders = [r for r in riders if r.on_duty]
        completed_orders = [o for o in orders if o.delivered]
        
        obs = f"当前有{len(active_riders)}个骑手在职，今日完成{len(completed_orders)}个订单"
        self.add_observation(obs)
        return obs
    
    def think(self, riders: List[Rider], environment_state: Dict) -> str:
        """分析决策"""
        unhealthy_riders = [r for r in riders if r.health < 3]
        complaints = sum(len(r.complaints) for r in riders)
        
        thought = f"有{len(unhealthy_riders)}个骑手健康状况不佳，收到{complaints}次投诉，需要考虑应对措施"
        self.add_thought(thought)
        return thought
    
    def calc_profit(self, orders: List[Order]) -> float:
        """计算日收益"""
        daily_orders = [o for o in orders if o.delivered]
        revenue = sum(o.cost * 0.8 for o in daily_orders)  # 平台抽成80%
        self.daily_revenue = revenue
        self.cash += revenue
        
        action = f"今日收益{revenue:.1f}元，累计资金{self.cash:.1f}元"
        self.add_action(action)
        return revenue
    
    def consider_fire_rider(self, rider: Rider) -> bool:
        """考虑是否解雇骑手"""
        if rider.health < 1 or len(rider.complaints) > 5:
            rider.on_duty = False
            action = f"解雇骑手{rider.agent_id}，原因：健康状况{rider.health:.1f}或投诉过多"
            self.add_action(action)
            return True
        return False
    
    def pay_tax(self, government) -> float:
        """向政府缴税"""
        tax_amount = self.daily_revenue * 0.1  # 10%税率
        self.cash -= tax_amount
        government.receive_tax(tax_amount)
        
        action = f"向政府缴税{tax_amount:.1f}元"
        self.add_action(action)
        return tax_amount

class Government(AgentState):
    """政府Agent"""
    
    def __init__(self):
        super().__init__("government", "Government")
        self.budget = 0.0  # 预算
        self.subsidies_paid = 0.0
        self.shelters_built = 0
        
    def observe(self, environment_state: Dict, riders: List[Rider]) -> str:
        """观察环境和骑手状况"""
        temp = environment_state["temperature"]
        complaints = sum(len(r.complaints) for r in riders)
        unhealthy_riders = len([r for r in riders if r.health < 5])
        
        obs = f"当前温度{temp:.1f}°C，收到{complaints}次投诉，{unhealthy_riders}个骑手健康状况不佳"
        self.add_observation(obs)
        return obs
    
    def think(self, environment_state: Dict, riders: List[Rider]) -> str:
        """思考政策决策"""
        temp = environment_state["temperature"]
        complaints = sum(len(r.complaints) for r in riders)
        
        if temp > 40 and complaints > 2:
            thought = "极端高温且投诉较多，需要采取紧急措施"
        elif temp > 38:
            thought = "高温天气，需要考虑预防措施"
        else:
            thought = "情况相对稳定，保持观察"
            
        self.add_thought(thought)
        return thought
    
    def decide_subsidy(self, environment_state: Dict, riders: List[Rider]) -> float:
        """决定高温补贴"""
        temp = environment_state["temperature"]
        
        if temp > 42:
            subsidy_per_rider = 50  # 极端高温补贴
        elif temp > 38:
            subsidy_per_rider = 30  # 高温补贴
        else:
            subsidy_per_rider = 0
            
        if subsidy_per_rider > 0:
            total_subsidy = 0
            for rider in riders:
                if rider.on_duty:
                    rider.money += subsidy_per_rider
                    total_subsidy += subsidy_per_rider
                    
            self.budget -= total_subsidy
            self.subsidies_paid += total_subsidy
            
            action = f"发放高温补贴，每人{subsidy_per_rider}元，总计{total_subsidy}元"
            self.add_action(action)
            return total_subsidy
        return 0
    
    def decide_build_shelter(self, environment_state: Dict, riders: List[Rider]) -> bool:
        """决定是否增设纳凉点"""
        temp = environment_state["temperature"]
        complaints = sum(len(r.complaints) for r in riders)
        shelter_rate = environment_state["shelter_rate"]
        
        # 高温且投诉多且覆盖率不够时考虑增设
        if temp > 40 and complaints > 3 and shelter_rate < 0.8:
            cost = 1000  # 增设纳凉点成本
            if self.budget >= cost:
                self.budget -= cost
                self.shelters_built += 1
                action = f"决定增设纳凉点，成本{cost}元"
                self.add_action(action)
                return True
        return False
    
    def receive_tax(self, amount: float):
        """接收税收"""
        self.budget += amount
        action = f"收到税收{amount:.1f}元，当前预算{self.budget:.1f}元"
        self.add_action(action)
