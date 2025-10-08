# Agent as tool pattern
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def research_agent_tool(query: str)-> str:
    """Research agent that provides market data and insights."""
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        model_info=ModelInfo(vision=True, function_calling=True, 
                             json_output=True,family="unknown", structured_output=True),
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    research_agent=AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message="""You are a specialist reseacrh agent.
        Provide concise and factual research findings."""
    )
    result=await research_agent.run(task="Research this topic:{query}")
    await model_client.close()
    return result.messages[-1].content

async def main()->None:
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        model_info=ModelInfo(vision=True, function_calling=True, 
                             json_output=True,family="unknown", structured_output=True),
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    coordinator=AssistantAgent(
        name="coordinator",
        model_client=model_client,
        tools=[research_agent_tool],
        system_message="""You are a project coordinator.
        Use research tools to provide comphrensive analysis"""
    )

    reviewer=AssistantAgent(
        name="reviewr",
        model_client=model_client,
        system_message="""You are a project reviewer.
        Evaluate the coordinator's analysis and suggest improvements."""
    )

    team=RoundRobinGroupChat(
        [coordinator,reviewer],
        termination_condition=MaxMessageTermination(6)
    )

    print("--AGENT-AS-TOOL PATTERN")
    result= await team.run(task="""Analyze the ROI of investing $100,000 in a Saas startup.
            Research the market and calculate potential returns.""")
    print(f"\nFinal analysis completed with {len(result.messages)} messages")

    print("Conversation transcript")
    for msg in result.messages:
        print(f"{msg.source}:{msg.content}\n")
    
    await model_client.close()

asyncio.run(main())