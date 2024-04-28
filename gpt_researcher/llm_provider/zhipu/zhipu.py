import os

from colorama import Fore, Style
from zhipuai import ZhipuAI

class ZhipuProvider:

    def __init__(self, model, temperature, max_tokens):
        self.model = model
        if (temperature == 0.0):
            temperature = 0.1
        self.temperature = temperature        
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.client = ZhipuAI(api_key=self.api_key)
    
    def get_api_key(self):
        try:
            api_key = os.environ["ZHIPU_API_KEY"]
        except:
            raise Exception(
                "Zhipu API key not found. Please set the ZHIPU_API_KEY environment variable.")
        return api_key
    
    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            chat = self.client.chat.completions.create(
                model="glm-3-turbo",
                messages=messages,
                temperature=self.temperature,
            )
            return chat.choices[0].message.content
        else:
            return await self.stream_response(messages, websocket)
    
    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        chat = self.client.chat.completions.create(
            model="glm-3-turbo",
            messages=messages,
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