#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) 演示
展示如何使用Langchain构建检索增强生成系统
"""

import os
import sys
from typing import List, Dict, Any
import logging

# 导入配置
from config import RAG_CONFIG, LLM_CONFIG, SAMPLE_DATA, setup_logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"请安装required packages: pip install langchain langchain-openai langchain-community faiss-cpu")
    sys.exit(1)

class RAGDemo:
    """RAG演示类"""
    
    def __init__(self, llm_type='qwen'):
        """初始化RAG系统
        
        Args:
            llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
        """
        self.logger = setup_logging()
        self.config = RAG_CONFIG
        self.llm_type = llm_type
        self.llm_config = LLM_CONFIG.get(llm_type, LLM_CONFIG['qwen'])
        
        # 初始化组件
        self.text_splitter = None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        self.logger.info(f"RAG系统初始化开始 - 使用模型: {llm_type}")
        self._setup_components()
        self.logger.info("RAG系统初始化完成")
    
    def _setup_components(self):
        """设置RAG组件"""
        try:
            # 1. 文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            self.logger.info(f"文本分割器设置完成 - chunk_size: {self.config['chunk_size']}")
            
            # 2. 嵌入模型
            if self.llm_type == 'qwen':
                # 对于千问，使用OpenAI兼容的embedding（通过阿里云接口）
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-v1",
                    openai_api_key=self.llm_config['api_key'],
                    openai_api_base=self.llm_config['base_url']
                )
            else:
                # 默认使用OpenAI embedding
                self.embeddings = OpenAIEmbeddings(
                    model=self.config['embedding_model'],
                    openai_api_key=self.llm_config['api_key']
                )
            self.logger.info(f"嵌入模型设置完成 - 类型: {self.llm_type}")
            
            # 3. 语言模型
            if self.llm_type == 'qwen':
                # 千问模型配置
                self.llm = ChatOpenAI(
                    model_name=self.llm_config['model_name'],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    openai_api_key=self.llm_config['api_key'],
                    openai_api_base=self.llm_config['base_url']
                )
            elif self.llm_type == 'ollama':
                # Ollama模型配置
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=self.llm_config['model_name'],
                    base_url=self.llm_config['base_url'],
                    temperature=self.llm_config['temperature']
                )
            else:
                # 默认OpenAI配置
                self.llm = ChatOpenAI(
                    model_name=self.llm_config['model_name'],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    openai_api_key=self.llm_config['api_key']
                )
            
            self.logger.info(f"语言模型设置完成 - model: {self.llm_config['model_name']}")
            
        except Exception as e:
            self.logger.error(f"组件设置失败: {str(e)}")
            raise
    
    def create_knowledge_base(self, documents: List[str]) -> None:
        """创建知识库"""
        try:
            self.logger.info("开始创建知识库")
            
            # 将文本转换为Document对象
            doc_objects = [Document(page_content=doc) for doc in documents]
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(doc_objects)
            self.logger.info(f"文档分割完成，共{len(split_docs)}个片段")
            
            # 创建向量存储
            self.vectorstore = FAISS.from_documents(
                documents=split_docs,
                embedding=self.embeddings
            )
            
            # 保存向量存储
            if not os.path.exists(self.config['vector_store_path']):
                os.makedirs(self.config['vector_store_path'])
            self.vectorstore.save_local(self.config['vector_store_path'])
            
            self.logger.info(f"知识库创建完成，保存至: {self.config['vector_store_path']}")
            
        except Exception as e:
            self.logger.error(f"知识库创建失败: {str(e)}")
            raise
    
    def load_knowledge_base(self) -> bool:
        """加载已有的知识库"""
        try:
            if os.path.exists(self.config['vector_store_path']):
                self.vectorstore = FAISS.load_local(
                    self.config['vector_store_path'],
                    self.embeddings
                )
                self.logger.info("知识库加载成功")
                return True
            else:
                self.logger.warning("知识库文件不存在")
                return False
        except Exception as e:
            self.logger.error(f"知识库加载失败: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """设置问答链"""
        try:
            if not self.vectorstore:
                raise ValueError("请先创建或加载知识库")
            
            # 自定义提示模板
            prompt_template = """
            基于以下上下文信息来回答问题。如果上下文信息不足以回答问题，请说"根据提供的信息无法回答该问题"。
            
            上下文信息:
            {context}
            
            问题: {question}
            
            详细回答:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # 创建检索器
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config['top_k']}
            )
            
            # 创建QA链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.logger.info("问答链设置完成")
            
        except Exception as e:
            self.logger.error(f"问答链设置失败: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """提问并获取答案"""
        try:
            if not self.qa_chain:
                raise ValueError("请先设置问答链")
            
            self.logger.info(f"收到问题: {question}")
            
            # 获取答案
            result = self.qa_chain({"query": question})
            
            # 整理结果
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    doc.page_content for doc in result["source_documents"]
                ]
            }
            
            self.logger.info(f"问题回答完成")
            return response
            
        except Exception as e:
            self.logger.error(f"问题回答失败: {str(e)}")
            return {
                "question": question,
                "answer": f"回答失败: {str(e)}",
                "source_documents": []
            }
    
    def similarity_search(self, query: str, k: int = None) -> List[str]:
        """相似性搜索"""
        try:
            if not self.vectorstore:
                raise ValueError("请先创建或加载知识库")
            
            k = k or self.config['top_k']
            docs = self.vectorstore.similarity_search(query, k=k)
            
            results = [doc.page_content for doc in docs]
            self.logger.info(f"相似性搜索完成，找到{len(results)}个相关文档")
            
            return results
            
        except Exception as e:
            self.logger.error(f"相似性搜索失败: {str(e)}")
            return []

def demo_rag(llm_type='qwen'):
    """RAG演示函数
    
    Args:
        llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
    """
    print(f"🚀 RAG (检索增强生成) 演示开始 - 使用模型: {llm_type}")
    print("=" * 50)
    
    try:
        # 初始化RAG系统
        rag = RAGDemo(llm_type=llm_type)
        
        # 尝试加载已有知识库，如果没有则创建新的
        if not rag.load_knowledge_base():
            print("📚 创建新的知识库...")
            rag.create_knowledge_base(SAMPLE_DATA['documents'])
        else:
            print("📚 加载已有知识库...")
        
        # 设置问答链
        rag.setup_qa_chain()
        
        # 演示相似性搜索
        print("\n🔍 相似性搜索演示:")
        search_query = "深度学习"
        similar_docs = rag.similarity_search(search_query)
        print(f"搜索查询: {search_query}")
        for i, doc in enumerate(similar_docs, 1):
            print(f"{i}. {doc}")
        
        # 演示问答
        print("\n❓ 问答演示:")
        for question in SAMPLE_DATA['questions']:
            print(f"\n问题: {question}")
            result = rag.ask_question(question)
            print(f"回答: {result['answer']}")
            if result['source_documents']:
                print("📄 相关文档:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"  {i}. {doc[:100]}...")
        
        print("\n✅ RAG演示完成!")
        
    except Exception as e:
        print(f"❌ RAG演示失败: {str(e)}")
        print("请检查配置和API密钥")

if __name__ == "__main__":
    demo_rag() 