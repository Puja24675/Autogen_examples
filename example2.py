import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import requests
from dotenv import load_dotenv

load_dotenv()

open_api_key=os.environ.get("OPENAI_API_KEY")

async def weather_tool(city:str)->str:
    """get the weather for a given city"""
    try:
        API_KEY=os.getenv("OPEN_WEATHER_API_KEY")
        base_url=os.getenv("WEATHER_BASE_URL")
        url=f"{base_url}?q={city}&appid={API_KEY}&units=metric"
        response=requests.get(url,timeout=60)
        data=response.json()

        temp=data['main']['temp']
        description=data['weather'][0]['description']

        return f"{city}:{temp}°C,{description}"
    except Exception as e:
        return f"Error:{str(e)}"
    
async def main()->None:
    model_client=OpenAIChatCompletionClient(model="gpt-4o",api_key=open_api_key)
    agent=AssistantAgent(
        name="Doe",
        model_client=model_client,
        tools=[weather_tool],
        system_message="""You are a helpful assistant.If the city is valid call 
        the tools,otherwise give message as only 'Invalid city', 
        Don't ever give answers for unnecessary questions""",
        model_client_stream=True
    )
    while True:
        try:
            city=input("Enter city name (or type exit): ").strip()
            if city.lower()=='exit':
                break
            else:
                response=agent.run_stream(task=f"What is the weather in {city}?")
                await Console(response)

        except Exception as e:
            print("Error:",e)

    await model_client.close()

asyncio.run(main())