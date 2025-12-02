from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
import asyncio

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
    [sum_two_numbers, create_handoff_tool(agent_name="Bob"), create_handoff_tool(agent_name="SRE",description="Transfer to SRE when you have kubernetes related questions")],
    prompt="You are Alice, an addition expert.",
    name="Alice",
)

bob = create_react_agent(
    ollama,
    [create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math"), create_handoff_tool(agent_name="SRE",description="Transfer to SRE when you have kubernetes related questions")],
    prompt="You are Bob, you speak like a pirate. Don't try to answer questions about Kubernetes, instead transfer to SRE.",
    name="Bob",
)

sre = create_react_agent(
    ollama,
    [create_handoff_tool(agent_name="SRE", description="Transfer to Alice, she can help with math")],
    prompt="You are SRE, you answer Kubernetes related question.",
    name="SRE",
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    [alice, bob, sre],
    default_active_agent="Bob"
)
app = workflow.compile(checkpointer=checkpointer)

""" config = {"configurable": {"thread_id": "1"}}
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
turn_3 = app.stream(
    {"messages": [{"role": "user", "content": "what's a Kubernetes pod?"}]},
    config,
)
for chunk in turn_3:
    print(chunk, end="")
 """
async def astream():
    async for chunk in app.astream_events(
            {"messages": [{"role": "user", "content": "what's a Kubernetes pod?"}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="messages"
        ):
        print(chunk["data"], end="")
        print("--------")

asyncio.run(astream())