import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import requests
from dotenv import load_dotenv

load_dotenv()

async def content_generator()->None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    content_writer_agent = AssistantAgent(
        name = "ContentGenerator",
        model_client = model_client,
        system_message = """You are a helpful content generator assistant. 
        Generate content for the given {topic} in 200 words.""",
        model_client_stream = True
    )

    content_reviewer_agent = AssistantAgent(
        name = "ContentReviewer",
        model_client = model_client,
        system_message = """You are a content reviewer agent. Consider given text 
        and review the content.Also give rating and score.""",
        model_client_stream = True
    )
    try:
        topic = input("Enter topic to generate content:")

        content_gen = await content_writer_agent.run(task=f"Topic is:{topic}")
        print("Content:",content_gen.messages[-1].content)

        content_rev = await content_reviewer_agent.run(
            task=f"Review this {content_gen.messages[-1].content}"
            )
        print("Content:",content_rev.messages[-1].content)

    except Exception as e:
        print("Error:",e)

    await model_client.close()

asyncio.run(content_generator())