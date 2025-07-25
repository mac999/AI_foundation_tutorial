import os, re, glob
import streamlit as st
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

VECTOR_DB_PATH = './faiss_index'
FILES_DIRECTORY = './files'
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# OpenAI 설정
llm_model = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def load_and_split_pdfs(files_directory):
    pdf_files = glob.glob(os.path.join(files_directory, '*.pdf'))
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_documents = splitter.split_documents(documents)
    for i, doc in enumerate(split_documents):
        print(f"Document {i}: {doc.page_content[:100]}...")
    return split_documents

def save_to_faiss(documents):
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    vectordb.save_local(VECTOR_DB_PATH)
    print(f"FAISS vector database saved to {VECTOR_DB_PATH}")
    return vectordb

def create_retrieval_qa(vectordb):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.25})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

def create_agent(qa_chain):
    tools = [
        Tool(
            name="Expert PDF File QA",
            func=qa_chain.run,
            description="질문에 대해 PDF 문서에서 답을 찾습니다."
        ),
        TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
    ]

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
        handle_parsing_errors=False
    )
    return agent

def extract_action_input(text):
    pattern = r'"action_input"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def clean_action_input_with_llm(text):
    prompt = f"""
다음 텍스트에서 진짜 대화 답변만 남기고, 시스템 메시지, 에러 메시지 등은 모두 삭제해줘.

텍스트:
{text}

지시사항:
- 대화형 답변 내용만 남기고, 에러 안내문은 삭제해라.
- 자연스럽게 이어지는 답변 문장만 남겨라.
- 추가 설명 없이 바로 정리된 답변만 출력해라.
- 포맷은 자연스러운 일반 문장으로 해라.

리턴할 결과:
"""
    response = llm_model.invoke(prompt)
    return response

def main():
    st.set_page_config(page_title="QA Expert Chatbot", page_icon="🤖", layout="wide")
    st.title("QA Expert Chatbot")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'agent' not in st.session_state:
        if not os.path.exists(VECTOR_DB_PATH):
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            docs = load_and_split_pdfs(FILES_DIRECTORY)
            save_to_faiss(docs)

        vectordb = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
        qa_chain = create_retrieval_qa(vectordb)
        agent = create_agent(qa_chain)
        st.session_state.agent = agent

    # 1. 채팅 기록 표시
    for user_msg, bot_msg in st.session_state.history:
        st.chat_message("user").markdown(user_msg)
        st.chat_message("assistant").markdown(bot_msg)

    st.markdown("---")

    # 2. 입력창
    user_input = st.text_input("💬 질문을 입력하세요:", key="input")

    # 3. 버튼 영역 (2개 버튼 나란히)
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("질문하기"):
            if user_input:
                try:
                    response = st.session_state.agent.run(user_input)
                    st.session_state.history.append((user_input, response))
                except Exception as e:
                    msg = f"Error: {str(e)}"
                    print(msg)

                    response = extract_action_input(str(e))
                    if not response:
                        response = clean_action_input_with_llm(str(e))

                    st.session_state.history.append((user_input, response))
                st.rerun()

    with col2:
        if st.button("대화 초기화"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()
