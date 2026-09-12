from dotenv import load_dotenv
from groq import Groq
from deepeval.models import DeepEvalBaseLLM
import os

load_dotenv()
class GroqModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM that uses the Groq API.
    """

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = "openai/gpt-oss-120b"

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the Groq chat completion API.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        Async version of the Groq generation method.
        """
        return self.generate(prompt)

    def get_model_name(self):
        """
        Return the name of the model used by this wrapper.
        """
        return self.model