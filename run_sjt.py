import json
from chatarena.arena import Arena
from chatarena.config import ArenaConfig
from chatarena.message import Message

def run_sjt():
    # 加载 SJT 配置文件
    sjt_config_path = r"E:\博士学习\LLM自适应测验\项目demo\chatarena\examples\sjt.json"
    with open(sjt_config_path, "r", encoding="utf-8") as f:
        sjt_config = json.load(f)
    Self_report = input("请输入自陈内容：")
    global_prompt = sjt_config["global_prompt"]
    moderator_config = {
        "role_desc": "",
        "backend": {
            "backend_type": "openai-chat",
            "temperature": 0.3,
            "max_tokens": 256
        }
        # moderator的大模型定义，这里用的openai
    }
    env_config = {
        "env_type": "SJT_env",
        # 创建游戏环境
        "parallel": False,
        # 是否并行发言
        "moderator": moderator_config,
        "moderator_visibility": "all",
        "moderator_period": "turn"
    }
    player_configs = []
    for i in range(len(sjt_config["players"])):
        player_name = f"Player {i + 1}"
        role_desc, backend_type, temperature, max_tokens = sjt_config["players"][i]["role_desc"], \
            sjt_config["players"][i]["backend"]["backend_type"], sjt_config["players"][i]["backend"]["temperature"], \
            sjt_config["players"][i]["backend"]["max_tokens"]
        player_config = {
            "name": player_name,
            "role_desc": role_desc,
            "backend": {
                "backend_type": backend_type,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
        }
        player_configs.append(player_config)

    arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config, global_prompt=global_prompt))

    arena.environment.message_pool.append_message(
        Message(agent_name="player 1", content=Self_report, turn=0, visible_to="player 1")# 只给 Player 1 拼接自陈内容
    )
    # 运行
    arena.run(num_steps=3)
    # 输出对话历史
    messages = arena.environment.get_observation()
    print("="*30 + " SJT 对话历史 " + "="*30)
    for msg in messages:
        print(f"第{getattr(msg, 'turn', '?')}轮 [{msg.agent_name}]：{msg.content}")
        # 如果是player的最终答案，且有反思轨迹，分开展示
        if hasattr(msg, 'agent_name') and msg.agent_name.lower().startswith('player'):
            # 尝试解析内容为元组（最终答案, round_records）
            if isinstance(msg.content, tuple) and len(msg.content) == 2:
                final_answer, round_records = msg.content
                print("【最终答案】:")
                print(final_answer)
                print("【反思过程】:")
                for r in round_records:
                    print(f"第{r['round']}轮反思：{r['reflection']}")
                    print(f"修正版答案：{r['revised_answer']}")
                    print("-" * 30)

    # 保存为txt
    with open("sjt_dialog_history.txt", "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"第{getattr(msg, 'turn', '?')}轮 [{msg.agent_name}]：{msg.content}\n")
            if hasattr(msg, 'agent_name') and msg.agent_name.lower().startswith('player'):
                if isinstance(msg.content, tuple) and len(msg.content) == 2:
                    final_answer, round_records = msg.content
                    f.write("【最终答案】：\n")
                    f.write(final_answer + "\n")
                    f.write("【反思过程】：\n")
                    for r in round_records:
                        f.write(f"第{r['round']}轮反思：{r['reflection']}\n")
                        f.write(f"修正版答案：{r['revised_answer']}\n")
                        f.write("-" * 30 + "\n")

if __name__ == "__main__":
    run_sjt()