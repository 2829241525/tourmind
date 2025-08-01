from mem0 import MemoryClient

client = MemoryClient(api_key="m0-IaS9rTU5d8K4Wy70HxcXcuVhX3CiIo3DizWP4xpb")
messages = [
    {"role": "assistant", "content": """
Dear Partner,
Well recieved with thanks!
The payment will be issued within this week.
I am copying our finance team's email: finance@tourmind.com for your reference. 
Thank you.
Looking forward to hearing from you.
Best Regards, 
Sofia Mo
"""},
    {"role": "user", "content": """Dear Sofia,
Dear Tourmind’s finance team,
We hope this email finds you all well.
Find below the current statement of your account with Juniper. We kindly would like to remind you that the payment of the 3rd instalment agreed for the API Develpment + Pack H2H + Supplier Tool A project should be arranged by 28th of February at the latest:"""},
]
resp = client.add(messages, user_id="u_email6", agent_id="a_email6")


# messages = [
#     {"role": "user", "content": "I'm travelling to San Francisco"},
#     {"role": "assistant", "content": "That's great! I'm going to Dubai next month."},
# ]
# resp = client.add(messages=messages, user_id="user1", agent_id="agent1")


print(resp)

# Example showing location and preference-aware recommendations
query = "邮件展示了哪些内容？"
filters = {
    "OR": [
        {
            "agent_id": "a_email1"
        },
        {
            "user_id": "u_email1"
        }
    ]
}
rsp = client.search(query, version="v2", filters=filters)
# print(rsp)

# filters = {
#     "AND": [
#         {
#             "user_id": "alex"
#         }
#     ]
# }

# all_memories = client.get_all(
#     version="v2", filters=filters, page=1, page_size=50)

# print(all_memories)
