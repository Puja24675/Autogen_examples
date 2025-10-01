import asyncio
# install autogen-agentchat,autogen-ext[openai]
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

open_api_key=os.environ.get("OPENAI_API_KEY")

async def main()->None:
    # create model client
    model_client=OpenAIChatCompletionClient(model="gpt-4o",api_key=open_api_key)
    # print(model_client)

    # create an assistant agent
    agent=AssistantAgent("assistant",model_client=model_client)
    response=await agent.run(task="say Hello world")
    print(response)
    print("========Response received==========")

    await model_client.close()

asyncio.run(main())