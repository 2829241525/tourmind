from mem0 import MemoryClient

client = MemoryClient(api_key="m0-IaS9rTU5d8K4Wy70HxcXcuVhX3CiIo3DizWP4xpb")


#user_id始终是你，metadata记录对方信息
messages_approach1 = [
    {"role": "user", "content": """
Dear Partner,
Well recieved with thanks!
The payment will be issued within this week.
I am copying our finance team's email: finance@tourmind.com for your reference. 
Thank you.
Looking forward to hearing from you.
Best Regards, 
Sofia Mo
"""},
    {"role": "user", "content": """
Dear Tourmind’s finance team,
We hope this email finds you all well.
Find below the current statement of your account with Juniper. We kindly would like to remind you that the payment of the 3rd instalment agreed for the API Develpment + Pack H2H + Supplier Tool A project should be arranged by 28th of February at the latest:"""},
]

# 使用metadata来记录邮件对方信息
# resp1 = client.add(
#     messages_approach1,
#     user_id="myself1",  # 始终是你自己
#     metadata={
#         "email_contact": "sofia@tourmind.com",
#         "contact_name": "Sofia Mo",
#         "company": "Tourmind",
#         "email_thread": "payment_reminder_api_project",
#         "project": "API Development + Pack H2H + Supplier Tool A"
#     }
# )
# print("方案1结果:", resp1)

# 方案2: 分别记录双方视角
# 从你的角度记录
resp2_your_view = client.add(
    [{"role": "user", "content": messages_approach1[0]["content"]}],
    user_id="xmx1@tourmind.com",
    metadata={
        "perspective": "outgoing_email",
        "recipient": "sofia@tourmind.com",
    }
)

# 从对方角度记录（作为你接收到的信息）
resp2_their_view = client.add(
    [{"role": "user", "content": messages_approach1[1]["content"]}],
    user_id="xmx1@tourmind.com",  # 使用对方作为user_id
    metadata={
        "perspective": "incoming_email",
        "sender": "sofia@tourmind.com",
    }
)

print("方案2 - 你的视角:", resp2_your_view)
print("方案2 - 对方视角:", resp2_their_view)

# 方案3: 使用对话ID统一管理
# conversation_id = "conversation_sofia_payment_2024"
# resp3 = client.add(
#     messages_approach1,
#     user_id="myself3",
#     metadata={
#         "conversation_id": conversation_id,
#         "participants": ["myself", "sofia@tourmind.com"],
#         "topic": "payment_reminder",
#         "date": "2024-02-28"
#     }
# )
# print("方案3结果:", resp3)

# 查询示例
print("\n=== 查询示例 ===")

# 查询所有与Sofia的邮件
# query1 = "与Sofia或Tourmind的邮件往来"
# filters1 = {
#     "AND": [
#         {"user_id": "myself1"},
#         {"metadata": {"email_contact": "sofia@tourmind.com"}}
#     ]
# }
# result1 = client.search(query1, version="v2", filters=filters1)
# print("与Sofia的邮件:", len(result1) if result1 else 0, "条")

# 查询所有付款相关邮件
query2 = "付款相关的邮件内容"
filters2 = {
    "AND": [
        {"user_id": "ar"}
    ]
}
result2 = client.search(query2, version="v2", filters=filters2)
print("付款相关邮件:", len(result2) if result2 else 0, "条")



# 查询特定项目邮件
# query3 = "API Development项目的邮件"
# result3 = client.search(query3, version="v2", user_id="myself3")
# print("API项目邮件:", len(result3) if result3 else 0, "条")
