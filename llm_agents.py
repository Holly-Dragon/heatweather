"""
集成LLM的Agent类
基于原有Agent结构，增加DeepSeek LLM智能决策能力
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import random
import uuid
import numpy as np

from agents import AgentState, Order  # 导入原有基础类
from llm_config import LLMEnhancedAgent, agent_prompts, deepseek_client
from environment import Environment

class LLMCustomer(LLMEnhancedAgent):
    """集成LLM的客户Agent"""
    
    def __init__(self, customer_id: str):
        super().__init__("Customer", customer_id)
        self.last_ratings = deque(maxlen=5)
        self.order_history = []
        self.last_order_hour = -99  # 记录上次下单时间，防止短时重复下单
        
    def observe_and_decide(self, environment: Environment, decision_mode: str = 'llm') -> Optional[Order]:
        """观察环境并做出点餐决策"""
        environment_state = environment.get_environment_state()
        hour = environment_state["hour"]
        is_meal_time = environment_state["is_meal_time"]

        # 检查是否在用餐时间
        if not is_meal_time:
            return None
        
        # 防止短时间内重复下单 (例如4小时内)
        if (hour - self.last_order_hour) < 4:
             return None

        temp = environment_state["temperature"]
        avg_rating = float(np.mean(self.last_ratings) if self.last_ratings else 5.0)
        
        # 构建观察信息
        observation = f"当前温度{temp:.1f}°C，时间{hour}点，用餐时间:{is_meal_time}，历史评分均值{avg_rating:.1f}"
        self.add_observation(observation)
        
        # 使用LLM进行决策
        if decision_mode == 'llm' and deepseek_client.is_available():
            user_prompt = f"""
当前环境:
- 温度: {temp:.1f}°C
- 时间: {hour}点 
- 是否用餐时间: {is_meal_time}
- 我的历史评分均值: {avg_rating:.1f}

请决定是否点外卖？考虑因素：
1. 高温对于外出就餐的便利程度
2. 用餐时间更适合点餐
3. 历史评分反映服务质量

请回答 {{"order": true/false, "concern_level": "high/medium/low", "reasoning": "决策理由"}}
"""
            
            decision = self.llm_decide(agent_prompts.CUSTOMER_SYSTEM, user_prompt)
            
            # 基于LLM决策结果
            should_order = False
            if decision.get("decision_type") == "rule_based":
                # 降级到规则决策
                should_order = self._rule_based_order_decision(temp, avg_rating, is_meal_time)
            else:
                # 解析LLM决策
                if isinstance(decision.get("order"), bool):
                    should_order = decision["order"]
                elif "order" in decision and decision["order"] in ["true", "True", True]:
                    should_order = True
                elif "reasoning" in decision:
                    # 从推理文本中判断意图
                    reasoning = decision["reasoning"].lower()
                    should_order = any(word in reasoning for word in ["点外卖", "订餐", "下单", "购买"])
                else:
                    should_order = self._rule_based_order_decision(temp, avg_rating, is_meal_time)
        else:
            # 使用规则决策
            self.add_thought("规则决策：基于温度、用餐时间和历史评分决定是否下单")
            should_order = self._rule_based_order_decision(temp, avg_rating, is_meal_time)
        
        # 如果决定下单，创建订单
        if should_order and is_meal_time:
            order = Order(
                order_id=str(uuid.uuid4()),
                customer_id=self.agent_id,
                time=hour,
                cost=environment.get_order_cost(),
                distance=environment.get_order_distance()
            )
            self.last_order_hour = hour  # 更新下单时间
            self.order_history.append(order)
            action = f"下单 - 订单ID:{order.order_id[:8]}, 金额:{order.cost:.1f}元, 距离:{order.distance:.1f}km"
            self.add_action(action)
            return order
        
        self.add_action("未下单")
        return None
    
    def _rule_based_order_decision(self, temp: float, avg_rating: float, is_meal_time: bool) -> bool:
        """规则基础的下单决策"""
        if not is_meal_time:
            return False
            
        temp_factor = max(0.1, 1.0 - (temp - 30) / 20)
        rating_factor = avg_rating / 5.0
        order_prob = temp_factor * rating_factor * 0.4
        
        return random.random() < order_prob
    
    def rate_order(self, order: Order, rider_health: float) -> int:
        """对订单评分"""
        base_rating = 5
        if rider_health < 5:
            base_rating -= 1
        if rider_health < 3:
            base_rating -= 1
            
        rating = max(1, min(5, base_rating + random.randint(-1, 1)))
        self.last_ratings.append(rating)
        order.rating = rating
        
        self.add_action(f"给订单{order.order_id[:8]}评分{rating}星")
        return rating
    
    def decide_tip(self, order: Order, rider_health: float) -> float:
        """决定小费"""
        tip = 0
        if rider_health < 3:
            tip = random.uniform(2, 5)  # 同情小费
        elif order.rating and order.rating >= 4:
            tip = random.uniform(1, 3)  # 满意小费
            
        order.tip = tip
        if tip > 0:
            self.add_action(f"给订单{order.order_id[:8]}小费{tip:.1f}元")
        return tip

class LLMRider(LLMEnhancedAgent):
    """集成LLM的骑手Agent"""
    
    def __init__(self, rider_id: str):
        super().__init__("Rider", rider_id)
        self.health = 10.0
        self.money = 1000.0
        self.on_duty = True
        self.happiness = 5.0
        self.orders_completed = 0
        self.complaints = []
        self.daily_income = 0.0
        
    def observe_and_decide(self, environment: Environment, available_orders: List[Order], decision_mode: str = 'llm') -> str:
        """观察环境并决定行动"""
        if not self.on_duty:
            return "off_duty"
            
        environment_state = environment.get_environment_state()
        temp = environment_state["temperature"]
        shelter_rate = environment_state["shelter_rate"]
        order_count = len(available_orders)
        
        # 构建观察信息
        observation = f"温度{temp:.1f}°C，阴凉覆盖率{shelter_rate:.1f}，可接订单{order_count}个，健康状况{self.health:.1f}/10"
        self.add_observation(observation)
        
        # 健康状况太差，强制休息
        if self.health < 2:
            self.add_action("健康状况危险，强制休息")
            return "rest"
        
        # 使用LLM进行决策
        if decision_mode == 'llm' and deepseek_client.is_available():
            system_prompt = agent_prompts.RIDER_SYSTEM.format(
                health=self.health,
                money=self.money,
                happiness=self.happiness
            )
            
            user_prompt = f"""
当前工作环境:
- 温度: {temp:.1f}°C (极端高温>40°C)
- 阴凉覆盖率: {shelter_rate:.1f}
- 可接订单数: {order_count}
- 我的健康状况: {self.health:.1f}/10
- 当前资金: {self.money:.0f}元

请选择最佳行动:
1. deliver - 接单配送 (有收入但可能损害健康)
2. rest - 休息恢复 (恢复健康但无收入)  
3. complain - 投诉工作条件 (表达不满)

请回答 {{"action": "deliver/rest/complain", "risk_level": "high/medium/low", "reasoning": "决策理由"}}
"""
            
            decision = self.llm_decide(system_prompt, user_prompt)
            
            # 解析LLM决策
            if decision.get("decision_type") == "rule_based":
                return self._rule_based_action_decision(temp, order_count)
            else:
                action = decision.get("action", "").lower()
                if action in ["deliver", "rest", "complain"]:
                    return action
                elif "deliver" in decision.get("reasoning", "").lower() and order_count > 0:
                    return "deliver"
                elif "rest" in decision.get("reasoning", "").lower():
                    return "rest"
                elif "complain" in decision.get("reasoning", "").lower():
                    return "complain"
                else:
                    return self._rule_based_action_decision(temp, order_count)
        else:
            self.add_thought("规则决策：基于健康、温度和订单数决定行动")
            return self._rule_based_action_decision(temp, order_count)
    
    def _rule_based_action_decision(self, temp: float, order_count: int) -> str:
        """规则基础的行动决策"""
        if self.health < 3 and random.random() < 0.3:
            return "complain"
        elif temp > 42 and self.health < 6 and random.random() < 0.7:
            return "rest"
        elif order_count > 0 and self.health >= 3:
            health_factor = self.health / 10.0
            temp_factor = max(0.1, 1.0 - (temp - 35) / 15)
            if random.random() < health_factor * temp_factor:
                return "deliver"
        
        return "rest"
    
    def deliver_order(self, order: Order, environment: Environment) -> Dict:
        """配送订单"""
        environment_state = environment.get_environment_state()
        temp = environment_state["temperature"]
        shelter_rate = environment_state["shelter_rate"]
        
        # 计算健康损失
        K = 0.2
        health_loss = (temp - 35) * K * order.distance * (1 - shelter_rate)
        health_loss = max(0, health_loss)
        
        self.health -= health_loss
        self.health = max(0, self.health)
        
        # 计算收入
        base_income = order.cost * 0.2
        total_income = base_income + order.tip
        self.money += total_income
        self.daily_income += total_income
        self.orders_completed += 1
        
        order.rider_id = self.agent_id
        order.delivered = True
        
        self.update_happiness(environment)
        
        action = f"完成订单{order.order_id[:8]}，收入{total_income:.1f}元，健康损失{health_loss:.1f}"
        self.add_action(action)
        
        return {
            "order": order,
            "income": total_income,
            "health_loss": health_loss
        }
    
    def rest(self, environment: Environment):
        """休息恢复"""
        environment_state = environment.get_environment_state()
        rest_rate = environment_state["rest_rate"]
        temp = environment_state["temperature"]
        
        recovery = 0.5 * rest_rate if temp > 40 else 1.0 * rest_rate
        self.health += recovery
        self.health = min(10, self.health)
        
        self.add_action(f"休息恢复健康{recovery:.1f}，当前健康{self.health:.1f}/10")
    
    def complain(self, environment: Environment):
        """投诉"""
        environment_state = environment.get_environment_state()
        complaint = {
            "rider_id": self.agent_id,
            "day": environment_state["day"],
            "hour": environment_state["hour"],
            "health": self.health,
            "temperature": environment_state["temperature"],
            "reason": "极端高温工作条件恶劣"
        }
        self.complaints.append(complaint)
        
        self.add_action(f"投诉工作条件，健康{self.health:.1f}/10，温度{environment_state['temperature']:.1f}°C")
        return complaint
    
    def update_happiness(self, environment: Environment):
        """更新幸福感"""
        environment_state = environment.get_environment_state()
        health_factor = self.health / 10.0
        money_factor = min(1.0, self.money / 2000.0)
        temp_factor = max(0.1, 1.0 - (environment_state["temperature"] - 30) / 20)
        
        self.happiness = (health_factor * 0.4 + money_factor * 0.3 + temp_factor * 0.3) * 10
        self.happiness = max(0, min(10, self.happiness))
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "rider_id": self.agent_id,
            "health": self.health,
            "money": self.money,
            "happiness": self.happiness,
            "on_duty": self.on_duty,
            "orders_completed": self.orders_completed,
            "daily_income": self.daily_income
        }

class LLMGovernment(LLMEnhancedAgent):
    """集成LLM的政府Agent"""
    
    def __init__(self):
        super().__init__("Government", "government")
        self.budget = 0.0
        self.subsidies_paid = 0.0
        self.shelters_built = 0
        
    def observe_and_decide(self, environment: Environment, riders: List[LLMRider], decision_mode: str = 'llm') -> Dict[str, Any]:
        """观察社会状况并制定政策"""
        environment_state = environment.get_environment_state()
        temp = environment_state["temperature"]
        complaints = sum(len(r.complaints) for r in riders)
        unhealthy_riders = len([r for r in riders if r.health < 5])
        shelter_rate = environment_state["shelter_rate"]
        
        observation = f"温度{temp:.1f}°C，{complaints}次投诉，{unhealthy_riders}个骑手健康不佳，覆盖率{shelter_rate:.2f}"
        self.add_observation(observation)
        
        # 使用LLM进行政策决策
        if decision_mode == 'llm' and deepseek_client.is_available():
            user_prompt = f"""
当前社会状况:
- 温度: {temp:.1f}°C
- 骑手投诉数: {complaints}
- 健康状况不佳骑手数: {unhealthy_riders}
- 纳凉点覆盖率: {shelter_rate:.2f}
- 政府预算: {self.budget:.0f}元

请制定政策措施：
1. 高温补贴：温度>38°C时每人30-50元，>42°C时50-100元
2. 增设纳凉点：投诉多且覆盖率<0.8时考虑，成本1000元
3. 紧急程度评估

请回答 {{"subsidy_amount": 数字, "build_shelter": true/false, "urgency": "high/medium/low", "reasoning": "政策理由"}}
"""
            
            decision = self.llm_decide(agent_prompts.GOVERNMENT_SYSTEM, user_prompt)
        else:
            self.add_thought("规则决策：基于温度、投诉和健康状况制定政策")
            decision = self._rule_based_policy_decision(temp, complaints, unhealthy_riders, shelter_rate)
        
        # 执行政策决策
        policies = {
            "subsidy": 0,
            "shelter": False,
            "reasoning": decision.get("reasoning", "基于当前状况的政策决策")
        }
        
        # 高温补贴
        subsidy_amount = decision.get("subsidy_amount", 0)
        if isinstance(subsidy_amount, (int, float)) and subsidy_amount > 0:
            if temp > 38:  # 只在高温时发放补贴
                total_subsidy = self._provide_subsidy(riders, subsidy_amount)
                policies["subsidy"] = total_subsidy
        
        # 增设纳凉点
        build_shelter = decision.get("build_shelter", False)
        if build_shelter and self.budget >= 1000 and shelter_rate < 0.8:
            if self._build_shelter():
                policies["shelter"] = True
        
        return policies
    
    def _rule_based_policy_decision(self, temp: float, complaints: int, unhealthy_riders: int, shelter_rate: float) -> Dict:
        """规则基础的政策决策"""
        decision = {"subsidy_amount": 0, "build_shelter": False, "urgency": "low"}
        
        if temp > 42:
            decision["subsidy_amount"] = 50
            decision["urgency"] = "high"
        elif temp > 38:
            decision["subsidy_amount"] = 30
            decision["urgency"] = "medium"
        
        if complaints > 3 and shelter_rate < 0.8:
            decision["build_shelter"] = True
            
        return decision
    
    def _provide_subsidy(self, riders: List[LLMRider], amount_per_rider: float) -> float:
        """发放高温补贴"""
        total_subsidy = 0
        for rider in riders:
            if rider.on_duty:
                rider.money += amount_per_rider
                total_subsidy += amount_per_rider
        
        self.budget -= total_subsidy
        self.subsidies_paid += total_subsidy
        
        self.add_action(f"发放高温补贴，每人{amount_per_rider}元，总计{total_subsidy}元")
        return total_subsidy
    
    def _build_shelter(self) -> bool:
        """增设纳凉点"""
        cost = 1000
        if self.budget >= cost:
            self.budget -= cost
            self.shelters_built += 1
            self.add_action(f"增设纳凉点，成本{cost}元")
            return True
        return False
    
    def receive_tax(self, amount: float):
        """接收税收"""
        self.budget += amount
        self.add_action(f"收到税收{amount:.1f}元，预算余额{self.budget:.1f}元")

class LLMPlatform(LLMEnhancedAgent):
    """集成LLM的平台Agent"""
    
    def __init__(self):
        super().__init__("Platform", "platform")
        self.cash = 5000.0
        self.daily_revenue = 0.0
        self.rider_pay_rate = 0.2
        
    def observe_and_decide(self, riders: List[LLMRider], orders: List[Order], decision_mode: str = 'llm') -> Dict[str, Any]:
        """观察运营状况并做决策"""
        active_riders = len([r for r in riders if r.on_duty])
        completed_orders = len([o for o in orders if o.delivered])
        avg_health = float(np.mean([r.health for r in riders]) if riders else 10.0)
        complaints = sum(len(r.complaints) for r in riders)
        
        observation = f"{active_riders}个骑手在职，完成{completed_orders}个订单，平均健康{avg_health:.1f}，{complaints}次投诉"
        self.add_observation(observation)
        
        # 使用LLM进行运营决策
        if decision_mode == 'llm' and deepseek_client.is_available():
            user_prompt = f"""
平台运营状况:
- 在职骑手数: {active_riders}
- 今日完成订单: {completed_orders}
- 骑手平均健康: {avg_health:.1f}/10
- 收到投诉数: {complaints}
- 平台资金: {self.cash:.0f}元

请做出运营决策：
1. 薪酬调整：可提高或降低骑手分成比例
2. 人员管理：是否解雇表现差的骑手
3. 风险评估：评估当前运营风险

请回答 {{"adjust_pay": 1.0, "fire_riders": false, "risk_level": "low", "reasoning": "决策理由"}}
"""
            
            decision = self.llm_decide(agent_prompts.PLATFORM_SYSTEM, user_prompt)
        else:
            self.add_thought("规则决策：基于骑手健康和投诉情况进行业务决策")
            decision = self._rule_based_business_decision(active_riders, avg_health, complaints)
        
        # 执行业务决策
        actions = {
            "pay_adjustment": 1.0,
            "fired_count": 0,
            "reasoning": decision.get("reasoning", "基于运营数据的决策")
        }
        
        # 薪酬调整
        pay_adjust = decision.get("adjust_pay", 1.0)
        if isinstance(pay_adjust, (int, float)) and pay_adjust != 1.0:
            self.rider_pay_rate *= pay_adjust
            actions["pay_adjustment"] = pay_adjust
            self.add_action(f"调整骑手分成比例至{self.rider_pay_rate:.2f}")
        
        # 解雇决策
        fire_riders = decision.get("fire_riders", False)
        if fire_riders:
            fired_count = self._fire_poor_performers(riders)
            actions["fired_count"] = fired_count
        
        return actions
    
    def _rule_based_business_decision(self, active_riders: int, avg_health: float, complaints: int) -> Dict:
        """规则基础的业务决策"""
        decision = {"adjust_pay": 1.0, "fire_riders": False, "risk_level": "low"}
        
        if avg_health < 4 or complaints > 5:
            decision["adjust_pay"] = 1.1  # 提高10%薪酬
            decision["risk_level"] = "high"
        elif avg_health < 6:
            decision["risk_level"] = "medium"
            
        if complaints > 10:
            decision["fire_riders"] = True
            
        return decision
    
    def _fire_poor_performers(self, riders: List[LLMRider]) -> int:
        """解雇表现差的骑手"""
        fired_count = 0
        for rider in riders:
            if rider.health < 1 or len(rider.complaints) > 5:
                rider.on_duty = False
                fired_count += 1
                
        if fired_count > 0:
            self.add_action(f"解雇{fired_count}名表现差的骑手")
        
        return fired_count
    
    def calc_profit(self, orders: List[Order]) -> float:
        """计算收益"""
        daily_orders = [o for o in orders if o.delivered]
        revenue = sum(o.cost * 0.8 for o in daily_orders)
        self.daily_revenue = revenue
        self.cash += revenue
        
        self.add_action(f"日收益{revenue:.1f}元，累计资金{self.cash:.1f}元")
        return revenue
    
    def pay_tax(self, government: LLMGovernment) -> float:
        """缴税"""
        tax_amount = self.daily_revenue * 0.1
        self.cash -= tax_amount
        government.receive_tax(tax_amount)
        
        self.add_action(f"缴税{tax_amount:.1f}元")
        return tax_amount
