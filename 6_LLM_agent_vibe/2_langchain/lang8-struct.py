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

# Step 1: Define Input Schema
class NewsInput(BaseModel):
    stock: str = Field(description="Stock ticker to get the latest news for")

    @validator('stock')
    def validate_stock(cls, v):
        if not v.isalpha() or len(v) > 5:
            raise ToolException('Stock ticker should only contain up to 5 alphabetic characters')
        return v.upper()

# Step 2: Define the tool function using Travily API
def get_company_news(stock: str) -> str:
    """Get the latest news about a stock using Travily API."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ToolException("Travily API key not set in environment variable `travily_api_key`")

    url = "https://travily.co.kr/api/news"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "query": stock
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ToolException(f"Failed to fetch news for {stock}. Status code: {response.status_code}")

    data = response.json()
    if not data.get("news"):
        return f"No news found for {stock}"

    top_news = data["news"][:5]
    formatted_news = "\n".join([f"- {item['title']} ({item['date']})" for item in top_news])
    return f"Latest news for {stock}:\n{formatted_news}"

# Step 3: Wrap with LangChain StructuredTool
get_company_news_tool = StructuredTool.from_function(
    func=get_company_news,
    args_schema=NewsInput,
    handle_tool_error=True,
)

# Step 4: LLM and prompt
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful financial assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [get_company_news_tool]

# Step 5: Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 6: Example run
result = agent_executor.invoke({"input": "Get me the latest news on Google (GOOGL)"})
print(result)
