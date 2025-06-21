"""
简化的LangGraph + LLM演示程序
用于测试DeepSeek集成和LangGraph工作流
"""

import os
import asyncio
import math
from typing import Dict, Any, List
from llm_config import DeepSeekClient, agent_prompts, check_llm_status
from environment import Environment

def test_deepseek_connection():
    """测试DeepSeek连接"""
    print("🔍 测试DeepSeek LLM连接...")
    
    client = DeepSeekClient()
    if not client.is_available():
        print("⚠️ DeepSeek API密钥未配置，使用模拟测试")
        return False
    
    # 测试API调用
    test_messages = [
        {"role": "system", "content": "你是一个测试助手"},
        {"role": "user", "content": "请用一句话介绍自己"}
    ]
    
    response = client.chat_completion(test_messages)
    if response:
        print(f"✅ DeepSeek连接成功!")
        print(f"📝 回应: {response[:100]}...")
        return True
    else:
        print("❌ DeepSeek连接失败")
        return False

async def test_llm_customer_decision():
    """测试LLM客户决策"""
    print("\n🧪 测试LLM客户决策")
    
    client = DeepSeekClient()
    
    # 模拟环境状态
    temp = 42.5
    is_meal_time = True
    hour = 12
    avg_rating = 4.2
    
    system_prompt = agent_prompts.CUSTOMER_SYSTEM
    user_prompt = f"""
当前环境:
- 温度: {temp:.1f}°C (极端高温!)
- 时间: {hour}点 (午餐时间)
- 是否用餐时间: {is_meal_time}
- 我的历史评分均值: {avg_rating:.1f}

考虑到极端高温可能影响骑手健康和配送质量，请决定是否点外卖？

请回答 {{"order": true/false, "concern_level": "high/medium/low", "reasoning": "决策理由"}}
"""
    
    if client.is_available():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(messages)
        if response:
            print(f"🤖 LLM客户分析:")
            print(f"   {response}")
        else:
            print("❌ LLM决策失败，使用规则决策")
    else:
        print("⚠️ 使用规则基础决策: 高温+用餐时间 = 50%概率下单")

async def test_llm_rider_decision():
    """测试LLM骑手决策"""
    print("\n🧪 测试LLM骑手决策")
    
    client = DeepSeekClient()
    
    # 模拟骑手状态
    health = 3.5
    money = 850
    happiness = 2.8
    temp = 45.2
    order_count = 3
    
    system_prompt = agent_prompts.RIDER_SYSTEM.format(
        health=health, money=money, happiness=happiness
    )
    
    user_prompt = f"""
当前工作环境:
- 温度: {temp:.1f}°C (危险高温!)
- 可接订单数: {order_count}
- 我的健康状况: {health:.1f}/10 (较差)
- 当前资金: {money:.0f}元

在健康不佳的情况下，面对危险高温和收入需求，请选择行动:
1. deliver - 接单配送 (收入但损害健康)
2. rest - 休息恢复 (恢复健康但无收入)
3. complain - 投诉工作条件 (表达不满)

请回答 {{"action": "deliver/rest/complain", "risk_level": "high", "reasoning": "决策理由"}}
"""
    
    if client.is_available():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(messages)
        if response:
            print(f"🤖 LLM骑手分析:")
            print(f"   {response}")
        else:
            print("❌ LLM决策失败")
    else:
        print("⚠️ 使用规则基础决策: 健康差+极端高温 = 优先休息")

async def test_llm_government_policy():
    """测试LLM政府政策决策"""
    print("\n🧪 测试LLM政府政策决策")
    
    client = DeepSeekClient()
    
    # 模拟社会状况
    temp = 43.8
    complaints = 7
    unhealthy_riders = 5
    shelter_rate = 0.3
    budget = 5000
    
    user_prompt = f"""
当前社会状况紧急！
- 温度: {temp:.1f}°C (极端高温预警)
- 骑手投诉数: {complaints} (较多)
- 健康状况不佳骑手数: {unhealthy_riders}
- 纳凉点覆盖率: {shelter_rate:.1f} (覆盖不足)
- 政府预算: {budget:.0f}元

作为负责劳动者保护的政府官员，请制定紧急政策措施：

1. 高温补贴建议：
   - 温度>38°C: 30-50元/人
   - 温度>42°C: 50-100元/人

2. 基础设施建设：
   - 增设纳凉点成本: 1000元/个
   - 投诉多且覆盖率低时优先建设

请回答 {{"subsidy_amount": 数字, "build_shelter": true/false, "urgency": "high", "reasoning": "政策理由"}}
"""
    
    if client.is_available():
        messages = [
            {"role": "system", "content": agent_prompts.GOVERNMENT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(messages)
        if response:
            print(f"🤖 LLM政府分析:")
            print(f"   {response}")
        else:
            print("❌ LLM决策失败")
    else:
        print("⚠️ 使用规则基础决策: 极端高温+多投诉 = 补贴70元+建纳凉点")

def simulate_simple_workflow():
    """模拟简单的工作流"""
    print("\n🔄 模拟简化工作流")
    
    # 环境状态
    env = Environment()
    env.current_hour = 12
    # 生成新的高温曲线来模拟极端情况
    env.temperature_curve = [35 + 10 * math.sin((h - 6) * math.pi / 12) + 5 for h in range(24)]
    env_state = env.get_environment_state()
    
    print(f"🌡️ 环境: {env_state['temperature']:.1f}°C, {env_state['hour']}点")
    
    # 1. 客户观察决策
    print("👤 客户观察环境...")
    if env_state['is_meal_time'] and env_state['temperature'] > 40:
        print("   🤔 午餐时间但高温，担心骑手健康")
        order_decision = "考虑到高温，暂不下单"
    else:
        order_decision = "正常下单"
    print(f"   📝 决策: {order_decision}")
    
    # 2. 骑手评估风险
    print("🚴 骑手评估工作环境...")
    rider_health = 4.2
    if env_state['temperature'] > 40 and rider_health < 5:
        action = "健康不佳，选择休息"
    else:
        action = "接单配送"
    print(f"   💭 健康: {rider_health}/10")
    print(f"   📝 决策: {action}")
    
    # 3. 政府监测情况
    print("🏛️ 政府监测社会状况...")
    if env_state['temperature'] > 40:
        policy = f"启动高温预警，发放补贴50元/人"
    else:
        policy = "正常监测"
    print(f"   📝 政策: {policy}")
    
    # 4. 平台调整策略
    print("💼 平台分析运营数据...")
    platform_action = "考虑到高温影响，暂时提高骑手分成比例"
    print(f"   📝 决策: {platform_action}")

async def main():
    """主测试程序"""
    print("🤖 LangGraph + DeepSeek LLM 集成测试")
    print("="*50)
    
    # 检查LLM状态
    check_llm_status()
    
    # 测试DeepSeek连接
    llm_available = test_deepseek_connection()
    
    if llm_available:
        print("\n🧠 测试LLM智能决策能力")
        await test_llm_customer_decision()
        await test_llm_rider_decision()
        await test_llm_government_policy()
    else:
        print("\n⚠️ LLM不可用，将使用规则基础决策")
    
    # 模拟工作流
    simulate_simple_workflow()
    
    print("\n" + "="*50)
    print("✅ 测试完成!")
    
    if llm_available:
        print("🎯 系统已准备好运行完整LangGraph+LLM仿真")
        print("💡 运行 python langgraph_simulation.py 开始完整仿真")
    else:
        print("💡 配置DeepSeek API密钥后可使用LLM智能决策")
        print("🔧 在.env文件中设置DEEPSEEK_API_KEY")

if __name__ == "__main__":
    asyncio.run(main())
