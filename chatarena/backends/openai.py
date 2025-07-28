import os
import re
from dotenv import load_dotenv
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend, register_backend
from ..message import  SYSTEM_NAME, Message, MessagePool, Question, QuestionPool
load_dotenv()
try:
    import openai
except ImportError:
    is_openai_available = False
    # logging.warning("openai package is not installed")
else:
    try:

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")  # 推荐用环境变量读取
        )
        is_openai_available = True
    except openai.OpenAIError:
        # logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-4o"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."


@register_backend
class OpenAIChat(IntelligenceBackend):
    """Interface to the ChatGPT style model with system, user, assistant roles separation."""

    stateful = False
    type_name = "openai-chat"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        merge_other_agents_as_one_user: bool = True,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=STOP,
        )

        response = completion.choices[0].message.content
        response = response.strip()
        return response

    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], ques: QuestionPool,
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ):
        """
        多轮反思与自我修正：每轮包括初步回答、反思、修正判断、修正版答案，最多5轮，直到模型判断无需修正。
        返回：final_answer, [每轮详细内容dict]
        """
        system_prompt = {"role": "system", "content": f"{global_prompt or ''}\n{role_desc}\n你是{agent_name}。"}
        conversations = []
        msgs = kwargs.get("msgs")
        if msgs:
            all_messages = msgs.get_all_messages()
            if all_messages:
                last_message = all_messages[-1]
                request_msg = last_message.content
        for msg in history_messages:
            role = "assistant" if msg.agent_name == agent_name else "user"
            conversations.append({"role": role, "content": f"{msg.agent_name}: {msg.content}{END_OF_MESSAGE}"})
        if request_msg:
            conversations.append({"role": "user", "content": f"{request_msg}{END_OF_MESSAGE}"})

        # 反思问题
        reflection_questions = ques.get_visible_questions(agent_name) if ques else []
        reflection_prompt = "请你根据以下自省问题进行反思，并逐条简要作答：\n"
        for q in reflection_questions:
            reflection_prompt += f"- {q.content}\n"
        reflection_prompt += "\n请逐条作答。"
        max_rounds = 3
        round_records = []
        current_answer = None
        for i in range(max_rounds):
            if i == 0:
                request = [system_prompt] + conversations + [{"role": "user", "content": "请根据以上信息，给出你的回答。"}]
                current_answer = self._get_response(request)
                current_answer = re.sub(rf"{END_OF_MESSAGE}$", "", current_answer).strip()
            # 2. 反思
            reflection_request = [system_prompt] + conversations
            reflection_request.append({"role": "assistant", "content": current_answer + END_OF_MESSAGE})
            reflection_request.append({"role": "user", "content": reflection_prompt})
            # 保存原温度，临时调高
            original_temperature = self.temperature
            self.temperature = 0.7
            reflection = self._get_response(reflection_request)
            self.temperature = original_temperature
            reflection = re.sub(rf"{END_OF_MESSAGE}$", "", reflection).strip()
            # 3. 直接修正（不再判断是否需要修正）
            revise_prompt = (
                "请根据你的反思，修正你的答案：\n"
                f"原答案：{current_answer}\n反思：{reflection}\n请输出修正版答案。"
            )
            revise_request = [system_prompt] + conversations
            revise_request.append({"role": "assistant", "content": current_answer + END_OF_MESSAGE})
            revise_request.append({"role": "assistant", "content": reflection + END_OF_MESSAGE})
            revise_request.append({"role": "user", "content": revise_prompt})
            revised_answer = self._get_response(revise_request)
            revised_answer = re.sub(rf"{END_OF_MESSAGE}$", "", revised_answer).strip()
            # 记录本轮内容
            round_records.append({
                "round": i+1,
                "answer": current_answer,
                "reflection": reflection,
                "revised_answer": revised_answer
            })
            current_answer = revised_answer
        final_answer = current_answer
        return final_answer, round_records
