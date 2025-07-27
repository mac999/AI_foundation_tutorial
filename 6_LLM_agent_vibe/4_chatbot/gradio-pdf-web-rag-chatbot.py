# pip install langchain gradio openai tavily-python pypdf faiss-cpu
# coding QA Expert Chatbot using langchain and gradio as web UI. use PDF RAG with faiss vector DB to save, retrieve the chunk documents from the PDF. if run this chatbot, read the PDF files from ./files folder, splite them into chunks, save them to faiss as vector database. after that, create LLM using openai and create langchain prompt template, tools with web search using Tavily. create agents with them including the previous dialog memory. this UI using gradio is simliar to ChatBot.
import os, re
import glob
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 1. 설정
CURRENT_FILE_PATH = os.path.abspath(__file__)
print(f"[INFO] Current file path: {CURRENT_FILE_PATH}")
VECTOR_DB_PATH = os.path.join(os.path.dirname(CURRENT_FILE_PATH), 'faiss_index')
FILES_DIRECTORY = os.path.join(os.path.dirname(CURRENT_FILE_PATH), 'files')

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# OpenAI 설정
llm_model = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# 2. PDF 파일 로드 및 벡터화
def load_and_split_pdfs(files_directory):
	pdf_files = glob.glob(os.path.join(files_directory, '*.pdf'))
	documents = []
	for file in pdf_files:
		loader = PyPDFLoader(file)
		documents.extend(loader.load())
	splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
	split_documents = splitter.split_documents(documents)
	for i, doc in enumerate(split_documents):
		print(f"Document {i}: {doc.page_content[:100]}...")  # Print the first 100 characters of each split document
	return splitter.split_documents(documents)

# 3. FAISS 벡터DB 저장
def save_to_faiss(documents):
	vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
	vectordb.save_local(VECTOR_DB_PATH)
	print(f"FAISS vector database saved to {VECTOR_DB_PATH}")
	return vectordb

# 4. RAG Retrieval QA 체인 생성
'''
Examples:

# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
	search_type="mmr",
	search_kwargs={'k': 6, 'lambda_mult': 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(
	search_type="mmr",
	search_kwargs={'k': 5, 'fetch_k': 50}
)

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
	search_type="similarity_score_threshold",
	search_kwargs={'score_threshold': 0.8}
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={'k': 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
	search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
Full name: langchain_core.vectorstores.base.VectorStore.as_retriever
'''
def create_retrieval_qa(vectordb):
	retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.25})
	qa_chain = RetrievalQA.from_chain_type(
		llm=llm_model,
		retriever=retriever,
		chain_type="stuff"
	)
	return qa_chain

# 5. LangChain Agent 생성
'''
Common agent Types:

zero-shot-react-description:
	Uses the ReAct (Reasoning + Acting) framework.
	Selects tools and generates responses based on tool descriptions.
	Best for scenarios where the agent needs to reason and act without prior context.

chat-zero-shot-react-description:
	Similar to zero-shot-react-description, but optimized for chat-based interactions.
	Useful for conversational agents.

chat-conversational-react-description:
	Designed for conversational agents with memory.
	Keeps track of the conversation history to provide context-aware responses.
	This is the agent type used in your code.

self-ask-with-search:
	Designed for agents that need to ask clarifying questions before answering.
	Often used with search tools.

react-docstore:
	Optimized for retrieving and reasoning over documents in a docstore.
	Useful for document-based question answering.

conversational-react-description:
	Similar to chat-conversational-react-description, but without explicit chat optimizations.
	Includes memory for context-aware responses.
'''
def create_agent(qa_chain):
	tools = [
		Tool(
			name="Expert PDF File QA",
			func=qa_chain.run,
			description="질문에 대해 PDF 문서에서 답을 찾습니다."
		),
		TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
	]

	'''
	prompt = PromptTemplate(
		input_variables=["input", "chat_history"],
		template="""
너는 친절한 코딩 Q&A 봇입니다. 지금까지의 대화는 다음과 같습니다:
{chat_history}

사용자의 질문:
{input}

적절한 도구를 사용해서 답하세요.
		"""
	)
	'''

	prompt = PromptTemplate(
		input_variables=["input", "chat_history"],
		template="""너는 친절하고 전문적인 코딩 Q&A 어시스턴트이다.

주어진 대화 내용을 참고하여 사용자의 질문에 대해 간결하고 명확한 답변을 작성하라.

- 오직 답변 내용만 작성하라.
- 서론, 결론, 불필요한 인삿말 없이, 질문에 대한 정확한 설명이나 해결책만 제공하라.

지금까지의 대화:
{chat_history}

사용자의 질문:
{input}

답변:
"""
	)


	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent = initialize_agent(
		tools=tools,
		llm=llm_model,
		agent="chat-conversational-react-description",
		memory=memory,
		verbose=True,
		agent_kwargs={"prompt": prompt},
		handle_parsing_errors=False # not working. 
	)
	return agent

def extract_action_input(text):
	# "action_input": "..." 패턴을 정규식으로 추출
	pattern = r'"action_input"\s*:\s*"([^"]+)"'
	match = re.search(pattern, text, re.DOTALL)
	if match:
		return match.group(1)  # 캡처한 "..." 안의 내용 리턴
	return None  

def clean_action_input_with_llm(text):
	prompt = f"""
다음 텍스트에서 "action_input" 값에 해당하는 부분만 정확히 추출해줘.
그 외의 모든 내용은 제거해. 

텍스트:
{text}

오직 "action_input" 안의 내용만 깔끔히 리턴해줘. 추가 설명 없이.
	"""

	response = llm_model.invoke(prompt)
	return response

# 6. Gradio UI
def main():
	# 초기화 과정
	if not os.path.exists(VECTOR_DB_PATH):
		os.makedirs(VECTOR_DB_PATH, exist_ok=True)
		docs = load_and_split_pdfs(FILES_DIRECTORY)
		save_to_faiss(docs)

	vectordb = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
	qa_chain = create_retrieval_qa(vectordb)
	agent = create_agent(qa_chain)

	def chatbot_interface(user_input, history):
		try:
			response = agent.run(user_input)
			history = history + [(user_input, response)]
		except Exception as e:
			msg = f"Error: {str(e)}"
			print(msg)

			response = extract_action_input(str(e))
			if response == None:
				response = clean_action_input_with_llm(str(e))
			history = history + [(user_input, response)]

		return history, history

	with gr.Blocks() as demo:
		gr.Markdown("Coding QA Expert Chatbot (PDF + Web Search)")
		chatbot = gr.Chatbot()
		msg = gr.Textbox(placeholder="질문을 입력하세요...") # 1) in pdf, what is b2gm? 2) i'm tom, developer. 3) who is tom

		clear = gr.Button("초기화")

		state = gr.State([])
		msg.submit(chatbot_interface, [msg, state], [chatbot, state])
		clear.click(lambda: ([], []), None, [chatbot, state])

	demo.launch()

if __name__ == "__main__":
	main()