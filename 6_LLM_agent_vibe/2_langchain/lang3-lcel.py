import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Step 1: ChatOpenAI 모델 초기화
key="YOUR_OPENAI_API_KEY"
llm = ChatOpenAI(temperature=0, openai_api_key=key)

# Step 2: JSON 형식 출력 템플릿 정의
json_prompt_template = """
Please provide the following details about {topic} in JSON format:
- Summary: A brief description of {topic}.
- Key Features: A list of important features of {topic}.
- Use Cases: A list of common use cases for {topic}.
- Related Technologies: A list of technologies related to {topic}.
"""

# ChatPromptTemplate 생성
json_prompt = ChatPromptTemplate.from_template(json_prompt_template)

# Step 3: 템플릿에 값 전달 (예: LangChain)
formatted_prompt = json_prompt.format(topic="LangChain")

# Step 4: LLMChain 생성
llm_chain = LLMChain(llm=llm, prompt=json_prompt)

# Step 5: LLMChain 실행
response = llm_chain.run(formatted_prompt)

# Step 6: JSON 형식으로 출력된 결과를 파싱
try:
    parsed_response = json.loads(response)  # 모델의 출력을 JSON으로 파싱
    print("Parsed JSON Output:")
    print(json.dumps(parsed_response, indent=4))
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("Raw response:", response)

