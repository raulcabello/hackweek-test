from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

ollama = ChatOllama(model="gpt-oss:20b", base_url="http://10.124.137.250:11434")


@tool
def sum_two_numbers(a: int, b: int) -> int:
    """Sum two numbers together.
    
    Args:
        a: The first number to sum
        b: The second number to sum
    
    Returns:
        The sum of a and b
    """
    return a + b


math_agent = create_agent(model=ollama, tools=[sum_two_numbers])
k8s_agent = create_agent(model=ollama, tools=[])

@tool(
    "math_agent",
    description="use this agent for math calculations",
)
def call_math_agent(query: str):
    result = math_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

@tool(
    "k8s_agent",
    description="use this agent for kubernetes related queries",
)
def call_k8s_agent(query: str):
    result = k8s_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    answer = result["messages"][-1].content
    return answer


agent = create_agent(model=ollama, tools=[call_math_agent, call_k8s_agent], system_prompt="you are supervisor agent that must always delegate to specialized agents. use the math_agent for math calculations and k8s_agent for kubernetes related queries.")

def main():
    query = "what is kubernetes?"
    for step in agent.stream({"messages": [{"role": "user", "content": query}]}):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()



if __name__ == "__main__":
    main()
