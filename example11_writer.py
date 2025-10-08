# Agent as tool pattern
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from typing import List,Literal
from dotenv import load_dotenv

load_dotenv()

class Story(BaseModel):
    title: str
    moral_of_story: str
    full_story: str

async def writer_agent_tool(topic: str)-> StructuredMessage:
    """Writer tool that generates story."""
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    writer_agent=AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message="""You are a creative writer agent.
        Generate story for given topic."""
    )

    result=await writer_agent.run(task=f"Generate story for this topic:{topic} in 60 words")
    await model_client.close()
    # print(result.messages[-1].content)
    story_text = result.messages[-1].content
    story = Story(
        title="Generated Title",  # You can extract this from story_text if needed
        moral_of_story="Extracted moral",  # Or use a placeholder
        full_story=story_text
    )
    return StructuredMessage(content=story)

async def main()->None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    coordinator=AssistantAgent(
        name="coordinator",
        model_client=model_client,
        tools=[writer_agent_tool],
        system_message="""You are a creative story writer coordinator.
        Use writer tools to provide good stories""",
        # output_content_type=Story
    )

    # reviewer=AssistantAgent(
    #     name="reviewer",
    #     model_client=model_client,
    #     system_message="""You are a project reviewer.
    #     Evaluate the coordinator's analysis and suggest improvements.""",
    # )

    team=RoundRobinGroupChat(
        [coordinator],
        termination_condition=MaxMessageTermination(6)
    )

    topic=input("Enter topic for generate: ")
    print("--AGENT-AS-TOOL PATTERN--")
    result= await team.run(task=f"Write a short story about {topic}.")
    
    if isinstance(result.messages[-1].content, StructuredMessage):
        story =result.messages[-1].content
        # print(story)
        print(f"\n--Title--:{story.title}")
        print(f"\n--Moral--:{story.moral_of_story}")
        print(f"\n--Story--:{story.full_story}")

    print(f"\n\n\nFinal story completed with {len(result.messages)} messages")

    # print("Conversation transcript")
    # for msg in result.messages:
    #     print(f"{msg.source}:{msg.content}\n")
    
    await model_client.close()

asyncio.run(main())