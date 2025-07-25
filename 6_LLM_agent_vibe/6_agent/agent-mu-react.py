import re, os
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Calculator function
def calculator(input_str: str) -> str:
    try:
        result = eval(input_str)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def FireCrawlResults(max_results=3, search_type="web"):
    from langchain_community.tools.fire_crawl import FireCrawlResults
    return FireCrawlResults(max_results=max_results, search_type=search_type) # TBD

# Initialize tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Evaluates mathematical expressions. Input should be a valid Python expression."
    ),
    TavilySearchResults(max_results=3, search_type="web"),
    # FireCrawlResults(max_results=3, search_type="web"),
]

# Function to extract tool prototype info
def get_tools_info(tools):
    info_list = []
    for tool in tools:
        # Try to get function signature if possible
        if hasattr(tool, "func"):
            proto = f"{tool.name}(input: str)"
        else:
            proto = f"{tool.name}(input: str)"
        desc = getattr(tool, "description", "No description.")
        info_list.append(f"- {proto}: {desc}")
    return "\n".join(info_list)

# Initialize the language model
llm = ChatOpenAI(temperature=0, model="gpt-4", max_tokens=4000)
answer_validation_llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", max_tokens=4000)

# Prompt template for ReAct
prompt_template = """
Instruction: {instruct}
IMPORTANT: If you do not know the answer, do not use 'Final Answer', just say 'I don't know'.

Tools you can use:
{tools}

Context:
{context}

Query:
{query}

You should follow the ReAct pattern:
- Think: Reason about the question or next step.
- Act: If needed, use a tool in the format Act: <tool>[<input>].
- Observe: Note the result of the action.
- Final: Give the final answer in the format Final Answer: <answer>.
"""

answer_validation_prompt = """
Question: {question}
Answer: {answer}
Is the answer relevant and correct for the question? Reply only with "yes" or "no" and a short reason.
"""

class ReActAgent:
    def __init__(self, llm, tools, prompt_template):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.prompt_template = prompt_template
        self.tools_info = get_tools_info(tools)

    def validate_answer(self, question, answer):
        prompt = answer_validation_prompt.format(question=question, answer=answer)
        messages = [HumanMessage(content=prompt)]
        output = answer_validation_llm(messages)
        response = output.content.strip().lower()
        return response.startswith("yes"), response

    def run(self, input_query: str):
        history = []
        while True:
            prompt = self.prompt_template.format(
                instruct="Answer the question in detail using ReAct reasoning.",
                tools=self.tools_info,
                context="\n".join(history),
                query=input_query
            )
            messages = [HumanMessage(content=prompt)]
            output = self.llm(messages)
            response = output.content.strip()
            history.append(response)

            # Check for Final Answer
            final_match = re.search(r"Final Answer:\s*(.*)", response, re.IGNORECASE)
            if final_match:
                answer = final_match.group(1)
                # is_valid, validation_msg = self.validate_answer(input_query, answer) # TBD. 실제 react 코드 확인.
                # if is_valid:
                return answer

            # Detect action
            action_match = re.search(r"Act:\s*(\w+)[\[\(](.*)[\]\)]", response)
            if action_match:
                action_name = action_match.group(1)
                action_input = action_match.group(2)
                print(f"Action detected: {action_name} with input: {action_input}")

                tool = self.tools.get(action_name)
                if not tool:
                    history.append(f"Observe: Unknown tool: {action_name}")
                    continue
                action_input = action_input.replace("'", "").replace('"', "")   
                result = tool.run(action_input)
                history.append(f"Observe: {result}")
            else:
                # If no action and no final answer, just continue (could be Think step)
                continue

agent = ReActAgent(llm=llm, tools=tools, prompt_template=prompt_template)

def test():
    query = "What is the result of 12 * (3 + 4)?"
    result = agent.run(query)
    print(result)

    query = "What is the capital of France?"
    result = agent.run(query)
    print(result)

    query = "In web, What is Taewook kang's paper about Geo BIM using BIM and GIS?"
    result = agent.run(query)
    print(result)

def main():
    print("ReAct Agent is ready to use.")
    print("Available tools:")
    for tool in agent.tools_info.split("\n"):
        print(tool)

    # test()

    print("\nInteractive mode:")
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = agent.run(query)
        print(result)

if __name__ == "__main__":
    main()

