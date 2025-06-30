# test a basic api request
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    # model="gpt-4.1-nano",
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a haiku about ai."}
    ]
)

print(response.choices[0].message.content)