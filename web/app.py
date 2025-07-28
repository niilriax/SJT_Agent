from flask import Flask, render_template, request
from chatarena.arena import Arena
from chatarena.config import ArenaConfig
from chatarena.message import Message
import os
import json

def run_sjt_web(self_report, reflection_on=True):
    SJT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../examples/sjt.json")
    with open(SJT_CONFIG_PATH, "r", encoding="utf-8") as f:
        sjt_config = json.load(f)
    global_prompt = sjt_config["global_prompt"]
    player_configs = []
    for i, player in enumerate(sjt_config["players"]):
        player_configs.append({
            "name": player["name"],
            "role_desc": player["role_desc"],
            "backend": player["backend"]
        })
    env_config = {
        "env_type": "SJT_env",
        "parallel": False,
        "moderator": {
            "role_desc": "",
            "backend": {
                "backend_type": "openai-chat",
                "temperature": 0.3,
                "max_tokens": 256
            }
        },
        "moderator_visibility": "all",
        "moderator_period": "turn"
    }
    arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config, global_prompt=global_prompt))
    # 插入自陈内容
    arena.environment.message_pool.append_message(
        Message(agent_name="Self_report", content=self_report, turn=0)
    )
    player1 = arena.players[0]
    observation = arena.environment.get_observation(player1.name)
    if reflection_on:
        final_response, round_records = player1.backend.query(
            agent_name=player1.name,
            role_desc=player1.role_desc,
            history_messages=observation,
            ques=arena.environment.question_pool,
            global_prompt=player1.global_prompt,
            msgs=arena.environment.message_pool
        )
    else:
        final_response, round_records = player1.backend.query(
            agent_name=player1.name,
            role_desc=player1.role_desc,
            history_messages=observation,
            ques=None,
            global_prompt=player1.global_prompt,
            msgs=arena.environment.message_pool
        )
    return final_response, round_records

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    self_report = ""
    reflection_on = True
    round_records = []
    if request.method == "POST":
        self_report = request.form.get("self_report", "")
        reflection_on = request.form.get("reflection_on") == "on"
        if self_report.strip():
            result, round_records = run_sjt_web(self_report, reflection_on)
    return render_template("index.html", result=result, self_report=self_report, reflection_on=reflection_on, round_records=round_records)

if __name__ == "__main__":
    app.run(debug=True) 