import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class BaseAgent(ABC):
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @abstractmethod
    def run(self, input_data):
        pass
