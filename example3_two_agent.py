import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import requests
from dotenv import load_dotenv

load_dotenv()

async def main()->None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    teacher=AssistantAgent(
        name="teacher",
        model_client=model_client,
        system_message="""You are a math teacher.Explain concepts clearly and ask follow-up 
        questions""",
        model_client_stream=True
    )

    student=AssistantAgent(
        name="teacher",
        model_client=model_client,
        system_message="""You are an eager student learning math.Ask questions when confused.""",
        model_client_stream=True
    )
    
    teacher_res=await teacher.run(task=f"If student ask any question takIf Explain what is probability to a beginner")
    print("Teacher:",teacher_res.messages[-1].content)

    student_res= await student.run(task=f"The teacher said:{teacher_res.messages[-1].content}.Please ask a clarifying question about probabilites.")
    print("Student:",student_res.messages[-1].content)
    await model_client.close()

asyncio.run(main())
