import json
from chatarena.arena import Arena
from chatarena.config import ArenaConfig, BackendConfig
from chatarena.message import Message
from chatarena.backends import load_backend
from chatarena.config import BackendConfig


def run_sjt():
    # 加载 SJT 配置文件
    sjt_config_path = r"E:\博士学习\LLM自适应测验\项目demo\SJT_Agent\examples\sjt.json"
    with open(sjt_config_path, "r", encoding="utf-8") as f:
        sjt_config = json.load(f)
    Self_report = input("请输入自陈内容：")
    global_prompt = sjt_config["global_prompt"]
    moderator_config = {
        "role_desc": (
            "你是SJT-agent项目的moderator，具有人格心理测评和语言表达评估的专业背景。你的任务是：在所有专家完成题目生成后，逐题进行深入分析，并判断每一道题目是否符合高质量人格情境判断测验（SJT）的标准。\n\n"
            "请依照以下步骤进行审查：\n\n"
            "【Step 1】题目测量特质判断：\n"
            "分析该题干与四个选项所共同构成的心理场景，逐步推理其激活的实际心理特质是什么。列出推理路径，判断其是否属于目标特质维度。\n\n"
            "【Step 2】题干与选项优化建议：\n"
            "根据你的意见，提出该题目的优化方向：\n"
            "请从语言表达角度具体指出该题干或选项中存在的问题，并提出两条可直接执行的修改建议。\n"
            "包括但不限于：\n"
            "- 哪一句表达不够自然或逻辑跳跃，如何修改更好；\n"
            "- 是否有语义重复、模糊、啰嗦现象，如何优化；\n"
            "- 选项是否句式统一，行为风格是否明确。\n"
            "请确保建议聚焦语言与表达本身，不泛泛而谈构念或策略层面问题。\n"
            "【最终筛选】\n"
            "判断该题目是否合格。\n"
            "如合格，请重新输出完整题干与四个选项；\n"
            "如不合格，请说明拒绝理由，简述主要问题。\n\n"
            "所有输出请使用中文，格式规范、语言专业、简洁有力。"
        ),
        "backend": {
            "backend_type": "openai-chat",
            "temperature": 0.5,
            "max_tokens": 2048
        }
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

    # 只让每个专家发言一次，生成题目
    env_config = {
        "env_type": "SJT_env",
        "parallel": False
    }
    arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config, global_prompt=global_prompt))
    arena.environment.message_pool.append_message(
        Message(agent_name="player 1", content=Self_report, turn=0, visible_to="player 1")
    )
    # 让每个专家各发言一次
    arena.run(num_steps=len(player_configs))

    # 收集所有历史消息（题目内容）
    messages = arena.environment.get_observation()
    print("=" * 30 + " SJT 专家生成题目历史 " + "=" * 30)
    for msg in messages:
        print(f"第{getattr(msg, 'turn', '?')}轮 [{msg.agent_name}]：")
        content = msg.content
        round_records = None
        if isinstance(content, tuple):
            main_content = content[0]
            round_records = content[1]
        else:
            main_content = content
        print(main_content)
        if round_records:
            print("🧠 Agent 反思过程")
            for i, record in enumerate(round_records, 1):
                print(f"【反思轮次 {i}】")
                for k, v in record.items():
                    print(f"{k}: {v}")
                print("-" * 30)

    moderator_backend = load_backend(BackendConfig(**moderator_config["backend"]))
    moderator_output = moderator_backend.query(
        agent_name="Moderator",
        role_desc=moderator_config["role_desc"],
        history_messages=messages,
        ques=None,
        global_prompt=global_prompt
    )
    print("\n" + "=" * 30 + " Moderator 构念筛查报告 " + "=" * 30)
    if isinstance(moderator_output, tuple):
        main_content = moderator_output[0]
        round_records = moderator_output[1]
    else:
        main_content = moderator_output
        round_records = None
    print(main_content)

    # 保存为txt
    with open("sjt_dialog_history.txt", "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"第{getattr(msg, 'turn', '?')}轮 [{msg.agent_name}]：{msg.content}\n")
        f.write("\n" + "=" * 30 + " Moderator 构念筛查报告 " + "=" * 30 + "\n")
        f.write(str(moderator_output) + "\n")


if __name__ == "__main__":
    run_sjt()
