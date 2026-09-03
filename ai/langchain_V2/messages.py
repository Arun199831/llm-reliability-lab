from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="What is Python commonly used for?")
]

response = model.invoke(messages)

print(response.content)

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools([get_word_length])

response = model_with_tools.invoke("How many letters are in the word 'transformer'?")

print("content:", response.content)
print("tool_calls:", response.tool_calls)


from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools([get_word_length])

messages = [HumanMessage(content="How many letters are in the word 'transformer'?")]

ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)

for call in ai_response.tool_calls:
    result = get_word_length.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final_response = model_with_tools.invoke(messages)
print(final_response.content)




 
