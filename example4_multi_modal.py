import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import Image
import PIL
import os
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

async def main()->None:
    model_client=OpenAIChatCompletionClient(model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))

    vision_agent=AssistantAgent(
        name="Vis",
        model_client=model_client,
        system_message="You are expert at descibing and analyzing images in detail"
    )
    # load image from web
    img_response=requests.get("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQshcDYQPTnlufRVdrYtkJakFSYMgOBcrcNTg&s")
    # img_response.content contains the image bytes
    # Bytesio is used to convert bytes data to a file-object that PIL can read
    pil_img=PIL.Image.open(BytesIO(img_response.content))
    # Convert into AutoGen image
    img=Image(pil_img)

    # we must fuse two inputs text and image so we used multimodalmessage
    multi_modal=MultiModalMessage(
        content=["Descibe this image in detail",img],
        source="user"
    )

    result= await vision_agent.run(task=multi_modal)
    print("Vision Analysis:",result.messages[-1].content)
    
    await model_client.close()



asyncio.run(main())