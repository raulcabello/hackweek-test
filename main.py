from supervisor import create_supervisor_agent
from swarn import create_swarn_agent
from command_parent import create_parent_agent, SubAgent
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.config import get_stream_writer
from langgraph.graph.state import CompiledStateGraph
from langchain.tools import tool

import asyncio


llm = ChatOllama(model="gpt-oss:20b", base_url="http://10.124.137.250:11434")
checkpointer = InMemorySaver()

@tool
def sum_two_numbers(a: int, b: int) -> int:
    """Sum two numbers together.
    
    Args:
        a: The first number to sum
        b: The second number to sum
    
    Returns:
        The sum of a and b
    """
    print("Summing", a, "and", b)
    return a + b


math_agent = create_agent(model=llm, tools=[sum_two_numbers])


AGENT_TYPE = "handoff" 

async def main():
    if AGENT_TYPE == "supervisor":
        agent = create_supervisor_agent()

    if AGENT_TYPE == "handoff":
        subragent = create_agent(model=llm, tools=[], system_prompt="You are a k8s expert.")
        k8s_agent = SubAgent(
            name="K8sAgent",
            description="Agent that can help with kubernetes related queries",
            agent=subragent
        )
        math_agent_sub = SubAgent(
            name="MathAgent",
            description="Agent that can help with math",
            agent=math_agent
        )
        agent = create_parent_agent(llm=llm, subagents=[k8s_agent, math_agent_sub], checkpointer=checkpointer)

    query = "what is a pod?"
    config = {"configurable": {"thread_id": "1"}}
    async for event in agent.astream_events({"messages": [{"role": "user", "content": query}]}, 
                             config=config,
                             stream_mode=["messages", "updates", "custom"]):
        if event["event"] == "on_chat_model_stream":
            if event["data"]["chunk"].content:
                print(event["data"]["chunk"].content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
