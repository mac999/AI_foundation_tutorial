from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI 

# Initialize the OpenAI LLM
api_key = ""

llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens\n')
    return result

# Create a conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

# Example usage
output = count_tokens(
    conversation, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)

print(output)
