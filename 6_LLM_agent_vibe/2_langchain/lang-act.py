import os
import requests
import json
from pydantic.v1 import BaseModel, Field, validator
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # assuming you are using OpenAI LLM

# Load environment variables from a .env file
load_dotenv()

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4", openai_api_key=key)
agent = initialize_agent(llm=llm, tools=[])  # 기본 에이전트 생성
query = "What is the weather today?"
response = agent.run(query)
print(response)
