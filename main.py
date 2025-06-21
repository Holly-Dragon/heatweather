"""
主程序入口
极端高温下外卖配送多主体仿真系统
"""

import sys
import os
from simulation import HeatWeatherSimulation

def main():
    """主函数"""
    print("=" * 60)
    print("极端高温下政府措施对外卖小哥幸福感健康程度影响仿真系统")
    print("基于LangGraph多主体Agent仿真")
    print("=" * 60)
    
    # 设置仿真参数
    try:
        print("\n请设置仿真参数（直接回车使用默认值）:")
        
        # 客户数量
        customers_input = input("客户数量 (默认10): ").strip()
        num_customers = int(customers_input) if customers_input else 10
        
        # 骑手数量
        riders_input = input("骑手数量 (默认3): ").strip()
        num_riders = int(riders_input) if riders_input else 3
        
        # 仿真天数
        days_input = input("仿真天数 (默认30): ").strip()
        simulation_days = int(days_input) if days_input else 30
        
        print(f"\n仿真配置:")
        print(f"- 客户数量: {num_customers}")
        print(f"- 骑手数量: {num_riders}")
        print(f"- 仿真天数: {simulation_days}")
        print(f"- 仿真总时长: {simulation_days * 24}小时")
        
        # 确认开始仿真
        confirm = input("\n确认开始仿真? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("仿真已取消")
            return
            
    except ValueError:
        print("参数输入错误，使用默认配置")
        num_customers = 10
        num_riders = 3
        simulation_days = 30
    except KeyboardInterrupt:
        print("\n仿真已取消")
        return
    
    # 创建仿真实例
    try:
        simulation = HeatWeatherSimulation(
            num_customers=num_customers,
            num_riders=num_riders,
            simulation_days=simulation_days
        )
        
        # 运行仿真
        results = simulation.run_simulation()
        
        # 输出结果摘要
        if results:
            print("\n" + "=" * 60)
            print("仿真结果摘要")
            print("=" * 60)
            
            print(f"仿真成功完成 {results.get('simulation_days', 0)} 天")
            print(f"订单完成率: {results.get('completion_rate', 0):.1%}")
            print(f"骑手平均健康水平: {results.get('avg_final_health', 0):.1f}/10")
            print(f"骑手平均幸福感: {results.get('avg_final_happiness', 0):.1f}/10")
            print(f"政府总补贴: {results.get('total_subsidies', 0):.0f}元")
            print(f"新建纳凉点: {results.get('shelters_built', 0)}个")
            
            # 政策效果评估
            print("\n政策效果评估:")
            avg_health = results.get('avg_final_health', 0)
            avg_happiness = results.get('avg_final_happiness', 0)
            
            if avg_health >= 7:
                health_assessment = "优秀"
            elif avg_health >= 5:
                health_assessment = "良好"
            elif avg_health >= 3:
                health_assessment = "一般"
            else:
                health_assessment = "较差"
                
            if avg_happiness >= 7:
                happiness_assessment = "优秀"
            elif avg_happiness >= 5:
                happiness_assessment = "良好"
            elif avg_happiness >= 3:
                happiness_assessment = "一般"
            else:
                happiness_assessment = "较差"
                
            print(f"- 健康水平: {health_assessment}")
            print(f"- 幸福感: {happiness_assessment}")
            
            # 建议
            print("\n政策建议:")
            if results.get('total_complaints', 0) > 10:
                print("- 投诉数量较多，建议增加高温补贴或改善工作条件")
            if results.get('avg_final_health', 0) < 5:
                print("- 骑手健康状况需要关注，建议增设更多纳凉点")
            if results.get('completion_rate', 0) < 0.8:
                print("- 订单完成率较低，建议平衡工作强度和健康保护")
            if results.get('final_shelter_rate', 0) < 0.5:
                print("- 纳凉点覆盖率仍需提升")
        
        print(f"\n详细日志已保存至: simulation_logs_day{simulation_days}.json")
        print("图表已保存至: simulation_results.png 和 rider_analysis.png")
        
    except Exception as e:
        print(f"\n仿真运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n仿真系统结束")

if __name__ == "__main__":
    main()
