import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import Literal, TypedDict, NotRequired
from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
import time

if not os.getenv("API_KEY"):
    raise ValueError("API_KEY environment variable is not set. Please set it to your OpenAI API key.")
llm = ChatOpenAI(
    model = os.getenv("MODEL", "step-1-8k"),
    api_key = os.getenv("API_KEY", ""),
    base_url = os.getenv("BASE_URL", ""),
)

# 1. 定义数据结构 (Pydantic Models)
# =======================================
class CustomerProfile(BaseModel):
    """顾客的个人信息"""
    customer_id: int = Field(..., description="顾客的唯一标识ID")
    age: int = Field(..., description="顾客的年龄")
    gender: Literal["男", "女", "其他"] = Field(..., description="顾客的性别")
    occupation: str = Field(..., description="顾客的职业，包括学生、教师、程序员、公务员、自由职业、演员、自媒体等")
    personality: str = Field(..., description="顾客的性格特点")
    economic_condition: str = Field(..., description="顾客的经济状况，包括：宽裕、普通、紧张")

class Order(BaseModel):
    """顾客的历史订单信息"""
    order_id: str = Field(..., description="订单的唯一标识ID")
    order_type: str = Field(..., description="订单的种类，例如：餐饮、生鲜、药品、文件")
    order_status: Literal["已完成", "已取消"] = Field(..., description="订单的状态")
    order_amount: float = Field(..., description="订单的金额")
    delivery_distance: float = Field(..., description="配送距离（公里）")
    delivery_time: str = Field(..., description="配送时长，例如：45分钟")
    order_rating: int = Field(..., description="订单的评分（1-5分）")
    order_review: str = Field(..., description="订单的评价内容")

class Customer(BaseModel):
    """完整的顾客信息，包含个人信息和历史订单"""
    profile: CustomerProfile = Field(..., description="顾客的个人信息")
    order_history: List[Order] = Field(..., description="顾客的历史订单列表")


# 2. 定义 Agent 的状态 (State)
# =======================================
class CustomerGenerationState(TypedDict):
    """
    Agent的状态，用于在图的节点之间传递数据。
    """
    # 要生成的顾客总数
    num_to_generate: int
    # 已生成的顾客信息列表，作为记忆模块
    past_customers: List[Customer]
    # 最新生成的顾客数据
    generated_customer: NotRequired[Customer]


# 3. 定义 Agent 的节点 (Nodes)
# =======================================
def generate_customer_node(state: CustomerGenerationState):
    """
    根据已有记忆生成一个新的、不重复的顾客信息。
    """
    print("---正在进入顾客生成节点---")
    
    past_customers_summary = []
    if state["past_customers"]:
        for customer in state["past_customers"]:
            past_customers_summary.append(f"- 职业: {customer.profile.occupation}, 年龄: {customer.profile.age}, 性别: {customer.profile.gender}")
    
    history_prompt = "你已经生成了以下顾客，请避免与他们过于相似：\n" + "\n".join(past_customers_summary) if past_customers_summary else "这是你要生成的第一个顾客。"

    prompt = f'''
    请你扮演一个富有创造力的数据模拟器。你的任务是生成一个**独特、具体且符合逻辑**的顾客信息。

    {history_prompt}

    请为这位顾客生成详细的个人画像和3条历史订单记录。

    **个人画像 (Profile):**
    *   **ID (customer_id)**: 请生成一个基于当前时间戳的整数ID。
    *   **年龄 (age)**: 请在18到60岁之间随机选择一个年龄。
    *   **性别 (gender)**: "男"、"女" 或 "其他"，请确保性别分布多样。
    *   **职业 (occupation)**: 请从一个广泛的职业列表中选择，并可以进行创造性发挥。例如：**花艺师、古董修复师、数据分析师、儿科医生、健身教练、独立游戏开发者、市场营销专员、餐厅服务员、退休干部**等。力求职业的多样性。
    *   **性格 (personality)**: 请用2-3个词描述其性格。例如：**精打细算、追求新潮、工作狂、养生达人、社交恐惧、美食家**。性格应与其职业和消费习惯相关。
    *   **经济条件 (economic_condition)**: "宽裕"、"普通" 或 "紧张"。

    **历史订单 (Order History):**
    *   订单内容(`order_type`)应与顾客的**职业、性格和经济条件**高度相关。
    *   订单的其他字段（金额、距离、评分、评价）也需要符合逻辑。

    请严格按照要求的JSON格式输出，确保数据充满多样性和真实感。
    '''
    
    structured_llm = llm.with_structured_output(Customer, method='function_calling')
    generated_customer = structured_llm.invoke(prompt)
    
    # 使用时间戳确保ID的独特性
    generated_customer.profile.customer_id = int(time.time() * 1000)

    print(f"---成功生成顾客: {generated_customer.profile.customer_id} (职业: {generated_customer.profile.occupation})---")
    
    # 更新状态，将新生成的顾客加入记忆
    return {
        "past_customers": state["past_customers"] + [generated_customer],
    }

def should_continue(state: CustomerGenerationState) -> Literal["continue", "end"]:
    """
    条件节点，判断是否继续生成。
    """
    if len(state["past_customers"]) < state["num_to_generate"]:
        print(f"---已生成 {len(state['past_customers'])}/{state['num_to_generate']}，继续生成...---")
        return "continue"
    else:
        print(f"---已生成 {len(state['past_customers'])}/{state['num_to_generate']}，任务结束。---")
        return "end"

# 4. 构建图 (Graph)
# =======================================
# 使用内存作为Checkpointer
memory = MemorySaver()
workflow = StateGraph(CustomerGenerationState)

workflow.add_node("generate_customer", generate_customer_node)
workflow.add_conditional_edges(
    "generate_customer",
    should_continue,
    {
        "continue": "generate_customer",  # 如果需要继续，则再次调用生成节点
        "end": END,                     # 否则结束
    },
)
workflow.set_entry_point("generate_customer")

# 编译图，并加入Checkpointer
app = workflow.compile(checkpointer=memory)

# 5. 运行 Agent
# =======================================
def save_customers_to_json(customers_list: List[Customer], file_path: str = './initdata/customer.json'):
    """
    将生成的顾客信息列表保存为JSON文件。

    Args:
        customers_list: 包含顾客Pydantic对象的列表。
        file_path: 要保存的JSON文件的路径。
    """
    print(f"\n--- 正在将生成的顾客信息保存到 {file_path} ---")
    try:
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 将Pydantic对象列表转换为字典列表
        customers_dict_list = [customer.dict() for customer in customers_list]
        
        # 写入JSON文件，使用 utf-8 编码以支持中文
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(customers_dict_list, f, indent=2, ensure_ascii=False)
            
        print(f"--- 成功保存 {len(customers_dict_list)} 位顾客信息到 {file_path} ---")
    except Exception as e:
        print(f"错误：保存文件时发生异常: {e}")

if __name__ == "__main__":
    NUMBER_OF_CUSTOMERS_TO_GENERATE = 3
    print(f"开始运行顾客生成器 Agent，目标生成 {NUMBER_OF_CUSTOMERS_TO_GENERATE} 位顾客...")

    # 配置一个唯一的会话ID，用于记忆
    config = {"configurable": {"thread_id": "customer-generation-thread"}}
    
    # 定义初始状态
    initial_state = {
        "num_to_generate": NUMBER_OF_CUSTOMERS_TO_GENERATE,
        "past_customers": [],
    }

    # 运行Agent
    final_state = app.invoke(initial_state, config=config)
    
    generated_customers_list = final_state.get('past_customers', [])

    if generated_customers_list:
        print("\n最终生成的全部顾客信息：")
        customers_dict_list = [customer.dict() for customer in generated_customers_list]
        print(json.dumps(customers_dict_list, indent=2, ensure_ascii=False))
        save_customers_to_json(generated_customers_list)



