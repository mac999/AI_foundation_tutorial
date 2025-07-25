from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경변수 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # API 키 설정

llm = ChatOpenAI(temperature=1.0, model='gpt-4o-mini')  

# LLM 응답 처리
def response(message, history, additional_input_info): # 사용자 입력 메시지, 이전 대화기록, 시스템 메시지
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content= additional_input_info))
    for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

# 인터페이스 생성
gr.ChatInterface(
    fn=response,   # LLM 응답처리 콜백함수 설정
    textbox=gr.Textbox(placeholder="Talk", container=False, scale=7),
    chatbot=gr.Chatbot(height=500),
    title="ChatBot",
    description="I'm a chatbot that can chat with you. I'm lovely chatbot.",
    theme="soft",
    examples=[["Hi"], ["I'm good"], ["What's your name?"]],
    additional_inputs=[
        gr.Textbox("", label="Input System Prompt", placeholder="I'm chatbot.") # placeholder="I'm chatbot." or "I'm a chatbot that can chat with you." or "I'm japanese chatbot."
    ]
).launch()