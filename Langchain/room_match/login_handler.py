#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
登录处理器 - 处理session过期和502错误的自动重新登录
"""

import requests
import json
import logging
import re
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LoginHandler:
    """登录处理器 - 只在502错误时重新登录"""

    # 登录配置
    LOGIN_CONFIG = {
        "url": "https://erp.tourmind.cn/api/login",
        "headers": {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json;charset=UTF-8',
            'Origin': 'https://erp.tourmind.cn',
            'Pragma': 'no-cache',
            'Priority': 'u=1, i',
            'Referer': 'https://erp.tourmind.cn/login',
            'Sec-Ch-Ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        },
        "credentials": {
            "UserName": "xiemingxuan",
            "Password": "xmx960313"
        },
        "initial_cookie": "admintools_user_session=MTc1MzY2Njg5M3xvSGpPeDAwTThTNHV1Wk5wWUhOSE5SOUYtTVlzMUlVUGJTZ2JXT0tNdWhacWNoRy1nQnR0cncwVHJlb1pSZFFESV9WdnFqYVg1UFU9fCuqoL1k-2_-FsqvBKEODcCSSEtPps-iCHUcbUck9mOy"
    }

    # Cookie存储配置
    COOKIE_STORAGE = {
        "env_var": "ADMIN_TOOLS_COOKIES",  # 环境变量名
        # 本地文件路径
        "file_path": os.path.join(os.path.dirname(__file__), ".admin_cookies")
    }

    def __init__(self):
        """初始化登录处理器"""
        self.current_cookies = self.load_stored_cookies()

    def load_stored_cookies(self) -> Optional[str]:
        """从存储位置加载cookies（优先级：环境变量 > 本地文件）"""
        try:
            # 1. 优先从环境变量读取
            env_cookies = os.environ.get(self.COOKIE_STORAGE["env_var"])
            if env_cookies and env_cookies.strip():
                logger.info("从环境变量加载cookies")
                return env_cookies.strip()

            # 2. 从本地文件读取
            file_path = self.COOKIE_STORAGE["file_path"]
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_cookies = f.read().strip()
                    if file_cookies:
                        logger.info("从本地文件加载cookies")
                        return file_cookies

            logger.info("未找到存储的cookies，将在首次登录时获取")
            return None

        except Exception as e:
            logger.error(f"加载存储的cookies失败: {str(e)}")
            return None

    def save_cookies(self, cookies: str):
        """保存cookies到存储位置（环境变量和本地文件）"""
        try:
            if not cookies:
                return

            # 1. 保存到环境变量（当前进程）
            os.environ[self.COOKIE_STORAGE["env_var"]] = cookies
            logger.info("已保存cookies到环境变量")

            # 2. 保存到本地文件（持久化）
            file_path = self.COOKIE_STORAGE["file_path"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cookies)
            logger.info(f"已保存cookies到本地文件: {file_path}")

        except Exception as e:
            logger.error(f"保存cookies失败: {str(e)}")

    def get_current_cookies(self) -> Optional[str]:
        """获取当前有效的cookies"""
        return self.current_cookies

    def extract_session_cookie(self, response_cookies: str) -> str:
        """从响应cookies中提取session信息"""
        try:
            # 提取admintools_user_session
            session_match = re.search(
                r'admintools_user_session=([^;]+)', response_cookies)
            session_cookie = session_match.group(1) if session_match else None

            # 提取acw_tc (如果存在)
            acw_match = re.search(r'acw_tc=([^;]+)', response_cookies)
            acw_cookie = acw_match.group(1) if acw_match else None

            # 构建完整的cookie字符串
            cookie_parts = []
            if acw_cookie:
                cookie_parts.append(f"acw_tc={acw_cookie}")
            if session_cookie:
                cookie_parts.append(
                    f"admintools_user_session={session_cookie}")

            return "; ".join(cookie_parts)

        except Exception as e:
            logger.error(f"提取session cookie失败: {str(e)}")
            return None

    def login_and_get_cookies(self) -> Optional[str]:
        """执行登录并获取新的cookies"""
        try:
            logger.info("开始执行登录获取新cookies...")

            # 准备登录请求
            headers = self.LOGIN_CONFIG["headers"].copy()
            # 使用当前cookies或初始cookie进行登录
            login_cookies = self.current_cookies or self.LOGIN_CONFIG["initial_cookie"]
            # headers['Cookie'] = login_cookies
            headers['cookie'] = login_cookies

            login_data = json.dumps(self.LOGIN_CONFIG["credentials"])

            # 执行登录请求
            response = requests.post(
                url=self.LOGIN_CONFIG["url"],
                headers=headers,
                data=login_data,
                verify=False
            )

            logger.info(f"登录请求完成，状态码: {response.status_code}")

            if response.status_code == 200:
                # 获取响应中的cookies
                response_cookies = response.headers.get('Set-Cookie', '')
                logger.info(f"登录响应cookies: {response_cookies}")

                # 提取并构建新的cookie字符串
                new_cookies = self.extract_session_cookie(response_cookies)

                if new_cookies:
                    self.current_cookies = new_cookies
                    # 自动保存新的cookies
                    self.save_cookies(new_cookies)
                    logger.info(f"登录成功，新cookies已保存: {new_cookies[:50]}...")
                    return new_cookies
                else:
                    logger.warning("登录响应中未找到有效的session cookie")
                    return None
            else:
                logger.error(
                    f"登录失败，状态码: {response.status_code}, 响应: {response.text}")
                return None

        except Exception as e:
            logger.error(f"登录过程中发生错误: {str(e)}")
            return None


# 全局登录处理器实例
login_handler = LoginHandler()


def get_login_handler() -> LoginHandler:
    """获取全局登录处理器实例"""
    return login_handler


if __name__ == "__main__":
    # 测试登录功能
    logging.basicConfig(level=logging.INFO)

    handler = LoginHandler()

    # 测试登录
    cookies = handler.login_and_get_cookies()
    if cookies:
        print(f"登录成功，获取到cookies: {cookies}")
    else:
        print("登录失败")
