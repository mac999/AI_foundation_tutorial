import os, getpass
from openai import OpenAI  
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_openai_functions_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

os.environ["TAVILY_API_KEY"] = "input your key"
os.environ["LANGCHAIN_PROJECT"] = "AGENT TUTORIAL"
load_dotenv()
client = OpenAI(api_key="input your key")

# Travily의 웹 검색 객체 획득
web_search = TavilySearchResults(k=5) 

# PDF 데이터를 벡터DB에 청크로 저장하고, 문서 검색 객체를 획득
loader = PyPDFLoader("./202212_LiDAR.pdf")  # 적절한 PDF 입력
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = loader.load_and_split(text_splitter)

embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
vector = FAISS.from_documents(split_docs, embeddings)
vectordb_retriever = vector.as_retriever()
output = vectordb_retriever.get_relevant_documents(
    "PCL(Point Cloud Library) 라이브러리에 대해 설명해줘"
)[0]
print(output)  # PCL 검색 예시

pdf_retriever_tool = create_retriever_tool(
    vectordb_retriever,
    name="pdf_search",
    description="2023년 12월 PCL(Point Cloud Library) 정보를 PDF 문서에서 검색합니다. '2023년 12월 라이다 포인트 클라우드 처리 기술' 과 관련된 질문은 이 도구를 사용해야 합니다!",
) # 문서 검색 객체

# 에이전트 도구들 설정
tools = [web_search, pdf_retriever_tool]

# LLM 객체 설정
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# 프롬프트 설정
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)

# LLM 함수 호출 에이전트 설정
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 전문가 에이전트 질문 수행
response = agent_executor.invoke(
    {
        "input": "2010년부터 PCL 라이브러리 기술에 대한 대한 내용을 PDF 문서에서 알려줘"
    }
)
print(f'답변: {response["output"]}')