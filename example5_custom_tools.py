import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import random
import math
from dotenv import load_dotenv

load_dotenv()

async def calculate_circle_area (radius:float) ->str:
    """Calculate the area of circle given it's radius"""
    print("Calling necessary tool")
    area=math.pi *radius**2
    return f"Area of circle of radius {radius} is {area} square units."

async def roll_dice (sides : int=6, count : int=1) ->str:
    """Roll dice and return the results"""
    print("Calling necessary tool")

    if count<1 or count>10:
        return "Can only roll between 1 and 10 dice at a time"
    if sides<2 or sides>100:
        return "Dice must have between 2 and 100 sides."
    
    results = [random.randint(1,sides) for _ in range(count)]
    total=sum(results)
    return f"Rolled {count}d{sides}:{results} (Total:{total})"  

async def get_random_fact()->str:
    """Get a random intresting fact."""
    print("Calling necessary tool")

    facts=[
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils. Achaeologists have found edible honey in ancient Egyptian tombs.",
        "A shrimp's heart is in its head.",
        "Bananas are berries, but strawberries aren’t",
    ] 
    return random.choice(facts)

async def main()->None:
    model_client=OpenAIChatCompletionClient(model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))

    tool_manager=AssistantAgent(
        name="Tom",
        model_client=model_client,
        tools=[calculate_circle_area,roll_dice,get_random_fact],
        system_message="""You are helpful assistant with access to various tools.use them to 
        help users with calculations,games,and intresting facts.You cannot access more than 
        2 tools per request.you don't have access to call more than 2 tools at a time.""",
        max_tool_iterations=3

    )
    # max_tool_iteration is atmost 3 iterations of tool calls before stopping the loop.
    # agent can be configured to execute multiple iterations until the model stops generating
        # tool calls or the maximum number of iterations is reached.

    tasks=[
        "Calculate area of circle with radius 5",
        "Roll 2 six-sided dice",
        "Give me a random intresting fact",
        "Calculate area of circle with radius 4 and roll 3 4-sided dice and an intresting fact"
    ]

    for task in tasks:
        print("Task: ",task)
        result= await tool_manager.run(task=task)
        print("Response:",result.messages[-1].content)

    await model_client.close()

asyncio.run(main())