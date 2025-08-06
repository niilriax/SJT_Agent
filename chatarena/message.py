import hashlib
import time
from dataclasses import dataclass
from typing import List, Union
from uuid import uuid1
from sentence_transformers import SentenceTransformer
import torch
import os
import sys
import re
# Preserved roles
SYSTEM_NAME = "System"
MODERATOR_NAME = "Moderator"


def _hash(input: str):

    hex_dig = hashlib.sha256(input.encode()).hexdigest()
    return hex_dig


@dataclass
class Message:
    agent_name: str
    content: Union[str, List[Union[str, int]]]
    turn: int
    timestamp: int = time.time_ns()
    visible_to: Union[str, List[str]] = "all"
    importance: int = 1
    msg_type: str = "text"
    logged: bool = False  # Whether the message is logged in the database
    embedding: torch.FloatTensor = torch.zeros((768,), dtype=torch.float32)
    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"agent: {self.agent_name}\ncontent: {self.content}\ntimestamp: {str(self.timestamp)}\nturn: {self.turn}\nmsg_type: {self.msg_type}"
        )


class MessagePool:
    def __init__(self):
        """Initialize the MessagePool with a unique conversation ID."""
        self.conversation_id = str(uuid1())
        self._last_message_idx = 0
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_qa = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        self.model_sym = SentenceTransformer('all-mpnet-base-v2')
        self._messages: List[Message] = []

    def save_exps_to(self, exps_path_to, current_game_number, is_incremental=False):
        if is_incremental:
            exps = [exp for exp in self._messages if exp.msg_type == "exp"]
            file_name = f"exps_{current_game_number}_incremental.pkl"
        else:
            exps = [exp for exp in self._messages if exp.msg_type == "exp" and exp.turn == current_game_number]
            file_name = f"exps_{current_game_number}_nonincremental.pkl"
        # 保存为txt
        file_name_txt = file_name + ".txt"
        with open(os.path.join(exps_path_to, file_name_txt), "w") as f:
            for exp in exps:
                f.write("Reflexion: " + str(exp.content[0]) + '\n')
                f.write("Talking content: " + str(exp.content[1]) + '\n')

    def reset(self):
        """Clear the message pool."""
        self._messages = []

    def give_importance(self, message: Message):
        content = message.content if message.msg_type in ("text", "ref") else message.content[0]
        if getattr(message, "agent_name", "") != "System" and getattr(message, "importance", 1) == 1:
            self_report_pattern = r"(我认为|我觉得|我会选择|我选择|我决定|我的理由|我的看法|我支持|我反对|我倾向于|我打算|我想|我希望|我建议|我不建议|我不认为|我不觉得)"
            reflection_pattern = r"(反思|总结|不足|改进|收获|遗憾|教训|启发|体会|经验)"
            if re.search(self_report_pattern, content) or re.search(reflection_pattern, content):
                message.importance = 5
                print(f"    give importance 5: {content}", file=sys.stderr)

    def append_message(self, message: Message):
        """
        Append a message to the pool.

        Parameters:
            message (Message): The message to be added to the pool.
        """
        self._messages.append(message)

    def append_message(self, message: Message):
        if hasattr(message, "importance") and message.importance == 0:
            return
        content = message.content if message.msg_type in ("text", "ref") else message.content[0]
        if message.msg_type in ("text", "ref"):
            message.embedding = torch.tensor(self.model_qa.encode(content), dtype=torch.float32)
        else:
            message.embedding = torch.tensor(self.model_sym.encode(content), dtype=torch.float32)
        #self.give_importance(message)
        self._messages.append(message)
        if getattr(message, "agent_name", "") == "Moderator":
            with open("moderator_log.md", "a", encoding="utf-8") as f:
                output = f"**{message.agent_name} (-> {str(message.visible_to)})**: {message.content}"
                f.write(output + "  \n")

    def append_message_at_index(self, message: Message, index: int):
        message.embedding = torch.from_numpy(self.model_qa.encode(message.content))
        self.give_importance(message)
        self._messages.insert(index, message)

    def print(self):
        """Print all the messages in the pool."""
        for message in self._messages:
            print(f"[{message.agent_name}->{message.visible_to}]: {message.content}")

    @property
    def last_turn(self):

        if len(self._messages) == 0:
            return 0
        else:
            return self._messages[-1].turn

    @property
    def last_message(self):

        if len(self._messages) == 0:
            return None
        else:
            return self._messages[-1]

    def get_all_messages(self) -> List[Message]:

        return self._messages

    def get_visible_messages(self, agent_name, turn: int) -> List[Message]:

        prev_messages = [message for message in self._messages if message.turn < turn]

        visible_messages = []
        for message in prev_messages:
            if (
                message.visible_to == "all"
                or agent_name in message.visible_to
                or agent_name == "Moderator"
            ):
                visible_messages.append(message)
        return visible_messages
@dataclass
class Question:
    content: str
    turn: int
    visible_to: str = 'all'
    reward: int = 0
    
    def __hash__(self):
        return int(self.msg_hash, 16)
    
    def __eq__(self, other):
        if isinstance(other, Message):
            return self.msg_hash == other.msg_hash
        return False

    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"content: {self.content}\nturn: {self.turn}\nvisible_to: {self.visible_to}")


class QuestionPool():
    def __init__(self):
        self._questions = self._initial_questions()
        self.conversation_id = str(uuid1())
        self._last_message_idx = 0
    
    @property
    def last_turn(self):
        if len(self._questions) == 0:
            return 0
        else:
            return self._questions[-1].turn

    def get_visible_questions(self, agent_name):

        return [que for que in self._questions if que.visible_to == agent_name]
    
    def get_necessary_questions(self):
        return [
            "我的职责和角色是什么? 我在这个任务中的最终目标是什么?",
            "基于合作规则，你能否知道其他两个agent的任务目标和职责?"
        ]
    
    def _initial_questions(self):
        # SJT特质分析专家反思问题
        questions = [
            Question(content="识别出的特质是否准确？为什么准确？给出理由", turn=0, visible_to="Player 1", reward=0),
            Question(content="是否遗漏了重要的行为倾向？", turn=0, visible_to="Player 1", reward=0),
            Question(content="你必须基于评分中发现的问题，明确提出两条可以优化的地方。每条建议需包括：1. 存在的问题是什么；2. 为什么会影响测验题目质量；3. 如何具体修改或提升。？", turn=0, visible_to="Player 1", reward=0),
        ]
        # SJT情景构建专家反思问题
        questions += [
            Question(
                content=(
                    "请你逐步判断构建的情景是否能够有效体现所要分析的特质：\n"
                    "Step 1：生成情景中意图测量的目标特质是什么；\n"
                    "Step 2：分析情景中包含的关键行为或决策线索；\n"
                    "Step 3：判断这些线索是否与目标特质高度相关，并说明理由。"
                ),
                turn=0,
                visible_to="Player 2",
                reward=0
            ),
            Question(
                content=(
                    "请你逐步评估情景是否具有典型性，能够代表目标特质在现实生活中的常见表现：\n"
                    "Step 1：分析情景是否贴近实际、具体清晰，为什么；\n"
                    "Step 2：判断该情境是否具有代表性，能否覆盖该特质在常见情境下的典型行为反应；\n"
                    "Step 3：若不具典型性，请指出原因并简述可能的改进方向。"
                ),
                turn=0,
                visible_to="Player 2",
                reward=0
            ),
            Question(
                content=(
                    "请你基于前面发现的问题，提出两条具体的优化建议。每条建议需包括以下三部分：\n"
                    "Step 1：指出当前存在的具体问题；\n"
                    "Step 2：说明该问题为何会降低测验题目的质量（例如可理解性、有效性、现实贴合度等）；\n"
                    "Step 3：提出清晰具体的修改或优化方式。"
                ),
                turn=0,
                visible_to="Player 2",
                reward=0
            ),
        ]
        # SJT选项匹配专家反思问题
        questions += [
            Question(
                content=(
                    "请你逐步判断每道题目的选项与情景是否匹配，能否构成完整的问题空间：\n"
                    "Step 1：特质与选项结合后所测量的心理特质是什么，为什么这么推测，理由是什么？；\n"
                    "Step 2：是否存在测量其他特质的可能性？如果不存在，说明理由；"
                ),
                turn=0,
                visible_to="Player 3",
                reward=0
            ),
            Question(
                content=(
                    "请你评估每个题目选项的梯度分布是否合理，是否能够体现行为强度或特质倾向的递进：\n"
                    "Step 1：梳理每个选项所体现的行为水平或心理倾向的强弱；\n"
                    "Step 2：判断选项之间是否呈现出清晰的梯度（从弱到强/从低到高）；\n"
                    "Step 3：指出是否存在梯度模糊或分布不均的问题，并说明可能的影响。"
                ),
                turn=0,
                visible_to="Player 3",
                reward=0
            ),
            Question(
                content=(
                    "请你分析选项中是否存在明显的正确答案，以及整体是否具有足够的区分度：\n"
                    "Step 1：检查是否有某一选项在道德性、逻辑或社会接受度上明显优于其他选项；\n"
                    "Step 2：判断选项之间是否能够真实反映不同水平的特质表达，而非倾向性过强或差异过小；\n"
                    "Step 3：指出是否影响受试者分化，并说明其对测验效果的潜在影响。"
                ),
                turn=0,
                visible_to="Player 3",
                reward=0
            ),
            Question(
                content=(
                    "请你基于上述反思中发现的问题，提出两条具体的优化建议。每条建议应包含以下三步：\n"
                    "Step 1：明确指出当前存在的问题；\n"
                    "Step 2：解释该问题会如何影响题目的区分度、信效度或测量一致性；\n"
                    "Step 3：提出具体、可实施的修改建议。"
                ),
                turn=0,
                visible_to="Player 3",
                reward=0
            ),
        ]

        # SJT协作反思问题
       # questions += [
        #    Question(content="我的输出是否与整个心理构念保持一致？", turn=0, visible_to="Moderator", reward=0),
        ##    Question(content="整体的SJT题目质量是否达到预期？", turn=0, visible_to="Moderator", reward=0),
        #]

        return questions

    def get_initial_questions(self, role):
        return [que for que in self._initial_questions if que.visible_to == role]