from mem0 import MemoryClient

client = MemoryClient(api_key="m0-IaS9rTU5d8K4Wy70HxcXcuVhX3CiIo3DizWP4xpb")
# messages = [
#     {"role": "user", "content": "我喜欢吃雪糕，雪糕分类有哪些？"},
#     {"role": "assistant", "content": """
# 雪糕宇宙分类​
# ​经典派​（安全牌，永不出错）
# ​香草​：奶香纯正，甜而不腻
# ​巧克力​：浓郁丝滑，治愈神器
# ​草莓​：酸甜清新，少女心狙击
# ​猎奇派​（胆大者入）
# ​芥末味​：上头直冲天灵盖
# ​臭豆腐味​：闻着臭，吃着…还是臭？
# ​辣椒巧克力​：冰火两重天
# ​奢华派​（贵但值得）
# ​黑松露冰淇淋​：一口几十块
# ​金箔冰淇淋​：吃的是仪式感
# ​茅台酒冰淇淋​：微醺吃冰新体验
# """},
# ]
# client.add(messages, user_id="test5", agent_id="test5")


messages = [
    {"role": "user", "content": "I'm travelling to San Francisco"},
    {"role": "assistant", "content": "That's great! I'm going to Dubai next month."},
]

resp = client.add(messages=messages, user_id="user1", agent_id="agent1")
print(resp)

# Example showing location and preference-aware recommendations
query = "我喜欢吃什么？雪糕有哪些分类？"
filters = {
    "OR": [
        {
            "agent_id": "agent1"
        },
        {
            "user_id": "user1"
        }
    ]
}
rsp = client.search(query, version="v2", filters=filters)
print(rsp)

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
