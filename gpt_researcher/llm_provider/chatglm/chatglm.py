import os
from colorama import Fore, Style
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3

from openai import OpenAI

class ChatGLMProvider:
    def __init__(self, model, temperature, max_tokens):
        self.model = "chatglm3-6b-32k"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint_url = self.get_endpoint_url()
        self.client = OpenAI(api_key="EMPTY", base_url=self.endpoint_url)

    def get_endpoint_url(self):
        try:
            endpoint_url = os.environ["CHATGLM3_ENDPOINT_URL"]
        except:
            raise Exception(
                "CHATGLM3_ENDPOINT_URL not found. Please set the CHATGLM3_ENDPOINT_URL environment variable.")
        return endpoint_url


    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            chat = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return chat.choices[0].message.content
        else:
            return await self.stream_response(messages, websocket)
    
    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in chat:
            content = chunk.choices[0].delta.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""
                    
        return response
