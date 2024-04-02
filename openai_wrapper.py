import os
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
import logging 
import json
import base64
from time import sleep

# Configure logging  
logging.basicConfig(level=logging.INFO)  
  
# Error callback function
def log_retry_error(retry_state):  
    logging.error(f"Retrying due to error: {retry_state.outcome.exception()}")  

DEFAULT_CONFIG = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.0,
    "max_tokens": 2048,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

class OpenAIWrapper:
    def __init__(self, config = DEFAULT_CONFIG, system_message="You are a helpful assistant."):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        print("set api key:", os.environ["OPENAI_API_KEY"])
        if os.environ.get("OPENAI_ORG_KEY") is not None:
            print("set organization key:", os.environ["OPENAI_ORG_KEY"])
            openai.organization = os.environ.get("OPENAI_ORG_KEY")

        # if os.environ.get("USE_AZURE")=="True":
        #     print("using azure api")
        #     openai.api_type = "azure"
        # openai.api_base = os.environ.get("API_BASE")
        # openai.api_version = os.environ.get("API_VERSION")

        self.config = config
        print("api config:", config, '\n')

        self.client = OpenAI()

        # count total tokens
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # system message
        self.system_message = system_message # "You are an AI assistant that helps people find information."

    # retry using tenacity
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3), retry_error_callback=log_retry_error)
    def completions_with_backoff(self, **kwargs):
        # print("making api call:", kwargs)
        # print("====================================")
        # return openai.ChatCompletion.create(**kwargs)
        return self.client.chat.completions.create(**kwargs)

    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3), retry_error_callback=log_retry_error)
    def assistant_w_code_interpreter_with_backoff(self, **kwargs):
        assistant = self.client.beta.assistants.create(
            name = self.config["assistant_name"],
            instructions = self.config["assistant_instruction"],
            tools=[{"type": "code_interpreter"}],
            model=self.config["model"]
        )
        thread = self.client.beta.threads.create()
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=kwargs["prompt"]
        )
        print("assistant message:", message)
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        max_tries = 50
        while True:
            retrieved_run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if retrieved_run.status == "completed":
                break
            print(f"waiting for completion on run_id {run.id}...")
            sleep(10)
            max_tries -= 1
            if max_tries == 0:
                return None, None
        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
        )
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id
        )
        return messages, run_steps


    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_input_message(self, text, images=[]):
        if images == []:
            return {"role":"user", "content":text}
        else:
            mes = {
                "role":"user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
            print("adding images:", images)
            for img_path in images:
                img = self.encode_image(img_path)
                mes["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                        # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    }
                })
            return mes

    def run(self, prompt, n=1, system_message="", images=[]):
        """
            prompt: str
            n: int, total number of generations specified
        """
        try:
            # overload system message
            if system_message != "":
                sys_m = system_message
            else:
                sys_m = self.system_message

            user_mes = self.get_input_message(prompt, images)
            if sys_m != "":
                # print("adding system message:", sys_m)
                messages = [
                    {"role":"system", "content":sys_m},
                    user_mes
                ]
            else:
                messages = [user_mes]
            print("input messages:", messages, "\n")

            text_outputs = []
            raw_responses = []
            rets = []

            while n > 0:
                cnt = min(n, 10) # number of generations per api call
                n -= cnt
                # print("messages:", messages, self.config)
                if self.config["api_type"] == "chat_completion":
                    chat_configs = {key: value for key, value in self.config.items() if key != "api_type"}
                    res = self.completions_with_backoff(messages=messages, n=cnt, **chat_configs)
                    text_outputs.extend([ {
                        "content": choice.message.content,
                        "role": choice.message.role,
                        "function_call": getattr(choice.message, "function_call", None),
                        "tool_calls": getattr(choice.message, "tool_calls", None),
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "index": getattr(choice, "index", None),
                    } for choice in res.choices])
                    # add prompt to log
                    ret = {
                        "id": res.id,
                        "model": res.model,
                        "prompt": prompt,
                        "system_message": sys_m,
                        "text_outputs": text_outputs
                    }
                    rets.append(ret)
                    raw_responses.append(res)
                    # log completion tokens
                    self.completion_tokens += res.usage.completion_tokens
                    self.prompt_tokens += res.usage.prompt_tokens
                elif self.config["api_type"] == "assistant_code_interpreter":
                    res_messages, run_steps = self.assistant_w_code_interpreter_with_backoff(prompt=prompt)
                    tool_calls = []
                    if run_steps is not None:
                        for run_step in run_steps.data:
                            if hasattr(run_step.step_details, "tool_calls"):
                                for tool_call in run_step.step_details.tool_calls:
                                    tool_calls.append(
                                        {
                                            "tool_type": "code_interpreter",
                                            "input": tool_call.code_interpreter.input,
                                            "output": tool_call.code_interpreter.outputs[0].logs
                                        }
                                    )
                        res = {
                            "messages": [{"content":res_messages.data[i].content[0].text.value, "role":res_messages.data[i].role} for i in range(len(res_messages.data))],
                            "tool_calls":tool_calls
                        }
                        ret = res['messages'][0]
                        rets.append(ret)
                        raw_responses.append(res)
                    else:
                        return [], []
            if len(rets) == 1:
                rets = rets[0]
            return rets, raw_responses

        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        model = self.config["model"]
        if model == "gpt-4-1106-preview":
            cost = self.completion_tokens / 1000 * 0.01 + self.prompt_tokens / 1000 * 0.03
        elif model == "gpt-3.5-turbo-1106":
            cost = self.completion_tokens / 1000 * 0.001 + self.prompt_tokens / 1000 * 0.002
        else:
            cost = 0 # TODO: add custom cost calculation for other engines
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}