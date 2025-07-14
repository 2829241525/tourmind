#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langchain项目主演示文件
整合工作流、Agent、RAG三大功能模块的演示
"""

import os
import sys
import logging
from datetime import datetime

# 导入配置
from config import setup_logging, PROJECT_ROOT

# 导入演示模块
try:
    from rag_demo import demo_rag
    from agent_demo import demo_agent
    from workflow_demo import demo_workflow
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖包已安装: pip install -r requirements.txt")
    sys.exit(1)

# ===== 全局配置 =====
DEMO_CONFIG = {
    'show_intro': True,
    'run_all_demos': True,
    'demo_timeout': 300,  # 每个演示的超时时间（秒）
    'interactive_mode': False  # 是否启用交互模式
}

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 Langchain 项目演示                      ║
    ║                                                              ║
    ║    展示 Langchain 在工作流、Agent、RAG 方面的基本使用         ║
    ║                                                              ║
    ║    作者: AI Assistant                                        ║
    ║    时间: {}                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(banner)

def print_menu():
    """打印功能菜单"""
    menu = """
    📋 可用演示功能:
    
    1. 🔄 Workflow (工作流演示)
       - 简单顺序工作流
       - 复杂顺序工作流
       - 自定义工作流
    
    2. 🤖 Agent (智能代理演示)
       - 多工具集成
       - 对话记忆
       - 复杂查询处理
    
    3. 📚 RAG (检索增强生成演示)
       - 知识库构建
       - 文档检索
       - 智能问答
    
    4. 🎯 All (运行所有演示)
    
    5. ❌ Exit (退出)
    """
    print(menu)

def setup_environment():
    """设置运行环境"""
    logger = setup_logging()
    logger.info("开始设置运行环境")
    
    # 检查项目目录
    if not os.path.exists(PROJECT_ROOT):
        os.makedirs(PROJECT_ROOT, exist_ok=True)
        logger.info(f"创建项目目录: {PROJECT_ROOT}")
    
    # 检查API密钥
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        logger.warning("⚠️  未设置有效的 OPENAI_API_KEY")
        print("⚠️  警告: 未检测到有效的 OpenAI API 密钥")
        print("   请设置环境变量: export OPENAI_API_KEY='your-api-key'")
        print("   或者修改 config.py 中的配置")
        print()
    
    logger.info("运行环境设置完成")
    return logger

def run_workflow_demo(llm_type='qwen'):
    """运行工作流演示"""
    print("\n" + "="*60)
    print(f"🔄 开始运行工作流演示 - 使用模型: {llm_type}")
    print("="*60)
    
    try:
        demo_workflow(llm_type=llm_type)
        print("\n✅ 工作流演示完成")
        return True
    except Exception as e:
        print(f"\n❌ 工作流演示失败: {str(e)}")
        return False

def run_agent_demo(llm_type='qwen'):
    """运行Agent演示"""
    print("\n" + "="*60)
    print(f"🤖 开始运行Agent演示 - 使用模型: {llm_type}")
    print("="*60)
    
    try:
        demo_agent(llm_type=llm_type)
        print("\n✅ Agent演示完成")
        return True
    except Exception as e:
        print(f"\n❌ Agent演示失败: {str(e)}")
        return False

def run_rag_demo(llm_type='qwen'):
    """运行RAG演示"""
    print("\n" + "="*60)
    print(f"📚 开始运行RAG演示 - 使用模型: {llm_type}")
    print("="*60)
    
    try:
        demo_rag(llm_type=llm_type)
        print("\n✅ RAG演示完成")
        return True
    except Exception as e:
        print(f"\n❌ RAG演示失败: {str(e)}")
        return False

def run_all_demos(llm_type='qwen'):
    """运行所有演示"""
    print(f"\n🎯 开始运行所有演示模块 - 使用模型: {llm_type}")
    print("="*60)
    
    results = {}
    
    # 运行工作流演示
    results['workflow'] = run_workflow_demo(llm_type)
    
    # 运行Agent演示
    results['agent'] = run_agent_demo(llm_type)
    
    # 运行RAG演示
    results['rag'] = run_rag_demo(llm_type)
    
    # 显示总结
    print("\n" + "="*60)
    print("📊 演示结果总结")
    print("="*60)
    
    for demo_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {demo_name.upper()}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 个演示成功完成")
    
    if success_count == total_count:
        print("🎉 所有演示都成功完成！")
    else:
        print("⚠️  部分演示失败，请检查配置和API密钥")

def interactive_mode():
    """交互模式"""
    logger = setup_logging()
    
    while True:
        try:
            print_menu()
            choice = input("\n请选择要运行的演示 (1-5): ").strip()
            
            if choice == '1':
                run_workflow_demo('qwen')
            elif choice == '2':
                run_agent_demo('qwen')
            elif choice == '3':
                run_rag_demo('qwen')
            elif choice == '4':
                run_all_demos('qwen')
            elif choice == '5':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-5")
            
            # 询问是否继续
            if choice in ['1', '2', '3', '4']:
                continue_choice = input("\n是否继续运行其他演示? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '是']:
                    print("👋 演示结束，再见！")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            logger.error(f"交互模式错误: {str(e)}")
            print(f"❌ 发生错误: {str(e)}")

def main():
    """主函数"""
    # 设置环境
    logger = setup_logging()
    logger.info("Langchain项目演示开始")
    
    # 显示横幅
    if DEMO_CONFIG['show_intro']:
        print_banner()
    
    # 设置运行环境
    setup_environment()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == 'workflow':
            run_workflow_demo('qwen')
        elif demo_type == 'agent':
            run_agent_demo('qwen')
        elif demo_type == 'rag':
            run_rag_demo('qwen')
        elif demo_type == 'all':
            run_all_demos('qwen')
        else:
            print(f"❌ 未知的演示类型: {demo_type}")
            print("可用选项: workflow, agent, rag, all")
            sys.exit(1)
    else:
        # 交互模式或运行所有演示
        if DEMO_CONFIG['interactive_mode']:
            interactive_mode()
        else:
            run_all_demos('qwen')
    
    logger.info("Langchain项目演示结束")
    print("\n🎉 感谢使用 Langchain 项目演示！")

if __name__ == "__main__":
    main() 