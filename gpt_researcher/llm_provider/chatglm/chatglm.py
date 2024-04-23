import os
from colorama import Fore, Style
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3


class ChatGLMProvider:
    def __init__(self, model, temperature, max_tokens):
        self.model = "ChatGLM3"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_endpoint_url(self):
        try:
            #endpoint_url = os.environ["CHATGLM3_ENDPOINT_URL"]
            endpoint_url = "http://edgegpu-proxy-003.gpumall.com:60161/v1/chat/completions"
        except:
            raise Exception(
                "CHATGLM3_ENDPOINT_URL not found. Please set the CHATGLM3_ENDPOINT_URL environment variable.")
        return endpoint_url


    def initialize_llm_chain(self, messages: list):
        template = "{input}"
        prompt = PromptTemplate.from_template(template)

        llm = ChatGLM3(
            endpoint_url=self.get_endpoint_url(),
            max_tokens=self.max_tokens,
            prefix_messages=messages,
            top_p=0.9
        )
        return LLMChain(prompt=prompt, llm=llm)

    def convert_messages(self, messages):
        """
        The function `convert_messages` converts messages based on their role into either SystemMessage
        or HumanMessage objects.
        
        Args:
          messages: It looks like the code snippet you provided is a function called `convert_messages`
        that takes a list of messages as input and converts each message based on its role into either a
        `SystemMessage` or a `HumanMessage`.
        
        Returns:
          The `convert_messages` function is returning a list of converted messages based on the input
        `messages`. The function checks the role of each message in the input list and creates a new
        `SystemMessage` object if the role is "system" or a new `HumanMessage` object if the role is
        "user". The function then returns a list of these converted messages.
        """
        converted_messages = []
        for message in messages:
            if message["role"] == "system":
                converted_messages.append(
                    SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                converted_messages.append(
                    HumanMessage(content=message["content"]))

        return converted_messages

    async def get_chat_response(self, messages, stream=False, websocket=None):
        system_messages = [message for message in messages if message["role"] == "system"]
        human_messages = [message for message in messages if message["role"] == "user"]
        llm_chain = self.initialize_llm_chain(messages=system_messages)
        if not stream:
            output = llm_chain.invoke({"input": human_messages})

            return output['text']

