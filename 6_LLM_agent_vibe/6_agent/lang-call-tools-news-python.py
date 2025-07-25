import re, requests
from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from bs4 import BeautifulSoup
from langchain.agents import tool
from langchain.agents import AgentExecutor
from dotenv import load_dotenv # API 키를 환경변수로 관리하기 위한 설정 파일
load_dotenv()

# 도구 생성
@tool
def search_news(news_url: str) -> str:
	"""Search naver.com news article and returns the body content."""
	try:
		headers = {
			"User-Agent": "Mozilla/5.0"
		}
		response = requests.get(news_url, headers=headers)
		if response.status_code == 200:
			soup = BeautifulSoup(response.text, "html.parser")
			
			# Find title with error handling
			content = soup.select_one("span.sds-comps-text.sds-comps-text-ellipsis-3.sds-comps-text-type-body1")

			if content:
				print(content.get_text(strip=True))
			else:
				print("해당 내용을 찾을 수 없습니다.")

			return content

		return f"HTTP error: {response.status_code}"
	except Exception as e:
		return f"error: {str(e)}"

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
	"""Use this to execute python code. If you want to see the output of a value,
	you should print it out with `print(...)`. This is visible to the user."""
	result = ""

	try:
		result = PythonREPL().run(code)

	except BaseException as e:
		print(f"Failed to execute. Error: {repr(e)}")

	return result


print(f"search: {search_news.description}")
print(f"python run tool: {python_repl_tool.description}")

prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a helpful assistant. "
			"Make sure to use the `search_news` tool for searching keyword related news.",
		),
		("placeholder", "{chat_history}"),
		("human", "{input}"),
		("placeholder", "{agent_scratchpad}"), # 에이전트의 스크래치 패드. 에이전트가 도구를 호출하기 전에 사용하는 공간입니다.
	]
)

# Agent 생성
tools = [search_news, python_repl_tool]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
	agent=agent,
	tools=tools,
	verbose=True,
	max_iterations=10,
	max_execution_time=10,
	handle_parsing_errors=True,
)

# 에이전트 실행
result = agent_executor.invoke({"input": "인공지능과 관련된 뉴스를 검색해 주세요."})
print(result["output"])

# 스트리밍 모드 실행. 에이전트가 작업을 수행하는 동안 각 단계의 결과를 실시간으로 반환함. 이 방식은 에이전트가 여러 도구를 호출하거나 복잡한 작업을 수행할 때, 중간 결과를 확인할 수 있도록 도와줌
result = agent_executor.stream({"input": "matplotlib 을 사용하여 pie 차트를 그리는 코드를 작성하고 실행하세요."}) 

for step in result:
	# 중간 단계 출력
	print(step)

