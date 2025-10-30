import os

from openai import AzureOpenAI, OpenAI


class OpenAIClient:
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def generate(
        self,
        prompt,
        temperature: float = 0.2,
        stop: list[str] = None,
        max_token: int | None = None,
    ):
        answer = str()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stop=stop,
                max_tokens=max_token,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print("Error: ", e)

        # Save to cache data
        return answer


class AzureOpenAIClient(OpenAIClient):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv(f"AZURE_ENDPOINT_{self.model.upper()}"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )
