import openai
from dotenv import load_dotenv
import os


load_dotenv()
class OPENAIClient:
    def __init__(self, model_name = "gpt-4o-mini", temperature = 0.5, top_p = 1):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key= api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, message, system_prompt):
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature= self.temperature,
            top_p= self.top_p
        )
        return response.choices[0].message.content

    