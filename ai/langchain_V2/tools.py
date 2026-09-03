import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model("gpt-4o-mini")

response = model.invoke("What is Python?")

print(response)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny"


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "What is the weather in New York?"
)

print(response)

for tool_call in response.tool_calls:
    print(f"tool: {tool_call['name']}")
    print(f"args: {tool_call['args']}")