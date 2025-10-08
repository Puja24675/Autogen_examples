import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main()->None:
    model_client = OpenAIChatCompletionClient(
        model = "gemini-2.5-flash",
        model_info = ModelInfo(vision=True, function_calling=True, 
                             json_output=True,family="unknown", structured_output=True),
        api_key = os.getenv("GOOGLE_API_KEY")
    )

    writer_agent = AssistantAgent(
        name = "creative_writer",
        model_client = model_client,
        system_message = """You are a creative writer specializing in science fiction.
        Write engaging stories with vivid descriptions and compelling characters.""",
        model_client_stream=True
    )

    result=await Console(
        writer_agent.run_stream(task="""Write a short science fiction story about a time
            traveler who discovers something unexpected about their past. 
            Make it engaging and include dialogue """),
        output_stats = True
    )
 
    await model_client.close()

asyncio.run(main())