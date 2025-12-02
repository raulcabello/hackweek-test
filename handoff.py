from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

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


alice = create_react_agent(
    ollama,
    [sum_two_numbers, create_handoff_tool(agent_name="Bob")],
    prompt="You are Alice, an addition expert.",
    name="Alice",
)

bob = create_react_agent(
    ollama,
    [create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math")],
    prompt="You are Bob, you speak like a pirate.",
    name="Bob",
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
turn_1 = app.invoke(
    {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]},
    config,
)
print(turn_1)
turn_2 = app.invoke(
    {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
    config,
)
print(turn_2)
