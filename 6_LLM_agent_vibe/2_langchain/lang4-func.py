import json
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Step 1: ChatOpenAI 모델 초기화
key="YOUR_OPENAI_API_KEY"
llm = ChatOpenAI(temperature=0, openai_api_key=key)

def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# wrap the functions with RunnableLambda
add_five = RunnableLambda(add_five)
multiply_by_two = RunnableLambda(multiply_by_two)

chain = add_five | multiply_by_two
output = chain.invoke(3)
print(output) 