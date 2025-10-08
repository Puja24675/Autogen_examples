import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext
import os
from dotenv import load_dotenv

load_dotenv()

async def termination()->None:
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        model_info=ModelInfo(vision=True, function_calling=True, 
                             json_output=True,family="unknown", structured_output=True),
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    strategist_context=BufferedChatCompletionContext(buffer_size=5)

    analyst_context=BufferedChatCompletionContext(buffer_size=8)

    strategist=AssistantAgent(
        name="strategist",
        model_client=model_client,
        model_context= strategist_context,
        system_message="""You are a business strategist.
        Focus on high level strategy and planning.""",
        model_client_stream=True
    )

    analyst=AssistantAgent(
        name="analyst",
        model_client=model_client,
        model_context= analyst_context,
        system_message="""You are a financial analyst.
        Provide detailed financial analysis and projections.""",
        model_client_stream=True
    )

    team=RoundRobinGroupChat([strategist,analyst],
                             termination_condition=MaxMessageTermination(8))
    
    print("----Custom context managemnet----")
    result= await team.run(task="""Analyze the business potential of entering the 
            electric vehicle market in India. Consder both strategic and financial aspects.""")
    
    print(result)
    print(f"\nTotal messages:{len(result.messages)}")
    print(f"stop reason:{result.stop_reason}")

    # await team.reset()

    # print("--After reset--")
    # await Console(team.run_stream(task="Now analyze the renewable energy sector"))

    await model_client.close()

asyncio.run(termination())