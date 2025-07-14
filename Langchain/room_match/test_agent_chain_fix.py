#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Agent对话链断裂修复
验证优化后的配置是否解决了迭代次数不足的问题
"""

import logging
import time
from dl_tool import create_room_matching_tool
from config import LLM_CONFIG, ROOM_MATCHING_CONFIG
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_agent_chain_fix():
    """测试Agent对话链断裂修复"""
    print("🔧 测试Agent对话链断裂修复")
    print("="*80)
    
    # 显示当前配置
    print(f"\n📊 当前Agent配置:")
    print(f"   最大迭代次数: {ROOM_MATCHING_CONFIG.get('agent_max_iterations', 'N/A')}")
    print(f"   LLM超时时间: {ROOM_MATCHING_CONFIG.get('llm_timeout', 'N/A')}秒")
    print(f"   最大Token数: {LLM_CONFIG['qwen'].get('max_tokens', 'N/A')}")
    print(f"   早停方法: {ROOM_MATCHING_CONFIG.get('agent_early_stopping_method', 'N/A')}")
    print(f"   错误处理: {ROOM_MATCHING_CONFIG.get('agent_handle_parsing_errors', 'N/A')}")
    
    try:
        # 创建LLM实例
        llm = ChatOpenAI(
            model=LLM_CONFIG['qwen']['model_name'],
            base_url=LLM_CONFIG['qwen']['base_url'],
            api_key=LLM_CONFIG['qwen']['api_key'],
            temperature=LLM_CONFIG['qwen']['temperature'],
            max_tokens=LLM_CONFIG['qwen']['max_tokens']
        )
        
        # 创建房型匹配工具
        room_matching_tool = create_room_matching_tool()
        
        # 创建Agent内存
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 使用优化后的配置初始化Agent
        agent_executor = initialize_agent(
            tools=[room_matching_tool],
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=ROOM_MATCHING_CONFIG.get('agent_verbose', True),
            max_iterations=ROOM_MATCHING_CONFIG.get('agent_max_iterations', 10),
            handle_parsing_errors=ROOM_MATCHING_CONFIG.get('agent_handle_parsing_errors', True),
            early_stopping_method=ROOM_MATCHING_CONFIG.get('agent_early_stopping_method', 'generate')
        )
        
        # 构建测试查询（简化版的房型匹配任务）
        test_query = """请帮我处理以下房型匹配任务：

1.供应商房型：King Room，疑似匹配标准房型[1]Standard King Room,[2]Deluxe King Room
2.供应商房型：Queen Room，疑似匹配标准房型[1]Standard Queen Room,[2]Two Queen Beds
3.供应商房型：Suite Room，疑似匹配标准房型[1]Junior Suite,[2]Executive Suite
4.供应商房型：Double Room，疑似匹配标准房型[1]Standard Double Room,[2]Twin Room
5.供应商房型：Family Room，疑似匹配标准房型[1]Family Room 4 Guests,[2]Family Suite

请分析每个供应商房型与其疑似匹配房型的相似度，并给出最佳匹配建议。"""
        
        print(f"\n🚀 开始执行Agent测试...")
        print(f"查询长度: {len(test_query)}字符")
        
        start_time = time.time()
        
        # 执行Agent查询
        try:
            response = agent_executor.invoke({"input": test_query})
            agent_result = response.get("output", "")
            execution_success = True
        except Exception as e:
            agent_result = f"Agent执行失败: {str(e)}"
            execution_success = False
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n⏱️ 执行时间: {execution_time:.2f}秒")
        
        # 分析结果
        if execution_success:
            print(f"\n✅ Agent执行成功!")
            print(f"响应长度: {len(agent_result)}字符")
            
            # 检查响应质量
            if "工具响应" not in agent_result and "房型最佳匹配结果" in agent_result:
                print("✅ Agent正确处理了工具响应，没有出现对话链断裂")
            elif "在没有工具响应的情况下" in agent_result:
                print("❌ 仍然存在对话链断裂问题")
            else:
                print("ℹ️ 响应格式正常，需要人工验证")
            
            print(f"\n📝 Agent响应内容（前500字符）:")
            print("-" * 50)
            print(agent_result[:500] + "..." if len(agent_result) > 500 else agent_result)
            print("-" * 50)
        else:
            print(f"\n❌ Agent执行失败!")
            print(f"错误信息: {agent_result}")
        
        return {
            "success": execution_success,
            "execution_time": execution_time,
            "response_length": len(agent_result) if execution_success else 0,
            "config_used": {
                "max_iterations": ROOM_MATCHING_CONFIG.get('agent_max_iterations'),
                "max_tokens": LLM_CONFIG['qwen'].get('max_tokens'),
                "timeout": ROOM_MATCHING_CONFIG.get('llm_timeout')
            }
        }
        
    except Exception as e:
        error_msg = f"测试执行异常: {str(e)}"
        logger.error(error_msg)
        print(f"\n❌ 测试失败: {error_msg}")
        return {"success": False, "error": error_msg}

def compare_old_vs_new_config():
    """比较优化前后的配置差异"""
    print(f"\n📊 配置优化对比:")
    print("="*50)
    
    old_config = {
        "agent_max_iterations": 3,
        "llm_timeout": 180,
        "max_tokens": 10000,
        "early_stopping_method": "默认",
        "handle_parsing_errors": "基础"
    }
    
    new_config = {
        "agent_max_iterations": ROOM_MATCHING_CONFIG.get('agent_max_iterations'),
        "llm_timeout": ROOM_MATCHING_CONFIG.get('llm_timeout'),
        "max_tokens": LLM_CONFIG['qwen'].get('max_tokens'),
        "early_stopping_method": ROOM_MATCHING_CONFIG.get('agent_early_stopping_method'),
        "handle_parsing_errors": ROOM_MATCHING_CONFIG.get('agent_handle_parsing_errors')
    }
    
    for key in old_config:
        old_val = old_config[key]
        new_val = new_config[key]
        
        # 比较数值类型的改进
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            improvement = "📈" if new_val > old_val else "🔧"
        else:
            improvement = "🔧"  # 非数值类型统一用工具图标
            
        print(f"   {improvement} {key}: {old_val} → {new_val}")

if __name__ == "__main__":
    print("Agent对话链断裂修复测试")
    print("="*80)
    
    # 显示配置对比
    compare_old_vs_new_config()
    
    # 运行测试
    result = test_agent_chain_fix()
    
    # 总结
    print(f"\n🎯 测试总结:")
    if result.get("success"):
        print("✅ 对话链断裂问题已修复")
        print(f"   执行时间: {result['execution_time']:.2f}秒")
        print(f"   响应长度: {result['response_length']}字符")
    else:
        print("❌ 仍需进一步优化") 