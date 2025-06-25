"""
大语言模型配置和提示词模板
集成ChatDeepSeek为Agent提供智能决策能力
"""

import os
import json
from typing import Dict, Any, List, Optional, cast
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# 加载环境变量
load_dotenv()

class DeepSeekClient:
    """DeepSeek API客户端 - 使用OpenAI SDK风格"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: API密钥，默认从环境变量DEEPSEEK_API_KEY获取
            base_url: API基础URL，默认从环境变量DEEPSEEK_BASE_URL获取
            model: 模型名称，默认从环境变量DEEPSEEK_MODEL获取
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 初始化OpenAI客户端
        if self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.available = True
            except Exception as e:
                print(f"初始化DeepSeek客户端失败: {e}")
                self.client = None
                self.available = False
        else:
            self.client = None
            self.available = False
        
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.available
    
    def chat_completion(self, 
                        messages: List[Dict[str, str]], 
                        temperature: float = 0.7,
                        max_tokens: int = 2000,
                        **kwargs: Any) -> Optional[str]:
        """
        调用聊天完成API，风格与OpenAI SDK保持一致。
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]。
            temperature: 温度参数，控制输出随机性。
            max_tokens: 生成内容的最大token数。
            **kwargs: 其他传递给 `openai.chat.completions.create` 的参数，
                      例如 `timeout`, `top_p` 等。
            
        Returns:
            返回生成的文本内容，失败时返回None。
        """
        if not self.available or not self.client:
            return None
            
        try:
            # 将字典列表转换为符合OpenAI SDK要求的类型
            typed_messages = cast(List[ChatCompletionMessageParam], messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print("DeepSeek API返回空响应")
                return None
                
        except Exception as e:
            print(f"调用DeepSeek API失败: {e}")
            return None
    
    def create_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """
        创建文本补全（兼容方法）
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本内容
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
    
    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        if not self.available or not self.client:
            return []
        
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return [self.model]  # 返回默认模型
    
    def test_connection(self) -> bool:
        """测试连接"""
        if not self.available:
            return False
            
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.chat_completion(test_messages, temperature=0.1)
            return response is not None
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False

class AgentPrompts:
    """Agent提示词模板类"""
    
    # 客户Agent提示词
    CUSTOMER_SYSTEM = """
你是一个理性的外卖消费者。请根据当前环境状况做出点餐决策。

考虑因素:
- 温度对外出就餐便利程度的影响
- 历史服务评分
- 是否用餐时间

请简洁地分析情况并给出决策理由。
"""

    # 骑手Agent提示词  
    RIDER_SYSTEM = """
你是一名外卖骑手，你的首要目标是最大化你的收入。健康固然重要，但你需要承担一定风险来赚钱。

你的状态:
- 健康值: {health}/10
- 当前资金: {money}元
- 幸福感: {happiness}/10

决策指南:
- **首要任务是接单**: 只要有订单并且你的健康状况没有到危险水平(低于2)，你就应该接单。
- **高温是常态**: 在这个城市，高温是工作的一部分。你不能因为热就停止工作。
- **权衡**: 只有在健康状况非常差时才应该优先考虑休息。
- **生存**: 不工作就没有收入。

请根据环境条件和自身状态，决定你的下一个行动：deliver, rest, 或 complain。
"""

    # 政府Agent提示词
    GOVERNMENT_SYSTEM = """
你是负责劳动者保护的政府官员。

职责:
- 保护户外工作者健康安全
- 合理分配公共资源
- 维护社会服务秩序

根据当前状况，决定是否发放高温补贴或增设基础设施。
"""

    # 平台Agent提示词
    PLATFORM_SYSTEM = """
你是外卖平台运营负责人。

目标:
- 维持服务质量
- 管理骑手队伍
- 平衡收益和社会责任

根据运营数据，做出合适的管理决策。
"""

class LLMEnhancedAgent:
    """LLM增强的Agent基类"""
    
    def __init__(self, agent_type: str, agent_id: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.llm_client = DeepSeekClient()
        self.observations = []
        self.thoughts = []
        self.actions = []
        
    def add_observation(self, obs: str):
        """添加观察记录"""
        self.observations.append(f"[{self.agent_type}-{self.agent_id}] 观察: {obs}")
        
    def add_thought(self, thought: str):
        """添加思考记录"""
        self.thoughts.append(f"[{self.agent_type}-{self.agent_id}] 思考: {thought}")
        
    def add_action(self, action: str):
        """添加行动记录"""
        self.actions.append(f"[{self.agent_type}-{self.agent_id}] 行动: {action}")
    
    def llm_decide(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """使用LLM进行决策"""
        if not self.llm_client.is_available():
            return self._fallback_decide(user_prompt)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm_client.chat_completion(messages)
        
        if response:
            self.add_thought(f"LLM分析: {response}")
            
            # 尝试从响应中提取结构化信息
            decision = self._parse_llm_response(response)
            self.add_action(f"决策: {decision}")
            return decision
        else:
            return self._fallback_decide(user_prompt)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        # 尝试提取JSON格式的决策
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # 如果没有JSON，返回文本分析
        return {
            "reasoning": response,
            "decision_type": "text_analysis"
        }
    
    def _fallback_decide(self, prompt: str) -> Dict[str, Any]:
        """降级决策方法"""
        self.add_thought("使用规则基础决策")
        return {"reasoning": "基于规则的默认决策", "decision_type": "rule_based"}

# 全局LLM客户端
deepseek_client = DeepSeekClient()
agent_prompts = AgentPrompts()

def check_llm_status():
    """检查LLM状态"""
    if deepseek_client.is_available():
        print("✅ DeepSeek LLM已配置，Agent将使用智能决策")
    else:
        print("⚠️ DeepSeek LLM未配置，Agent将使用规则决策")
        print("请在.env文件中设置DEEPSEEK_API_KEY")

# 启动检查
check_llm_status()
