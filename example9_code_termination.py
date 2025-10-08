import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def termination()->None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    code_writer_agent=AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message="""You are a code generator. Generate code based on requests 
        and improve it based on feedback""",
        model_client_stream=True
    )

    code_feedback_agent=AssistantAgent(
        name="Feedback",
        model_client=model_client,
        system_message="""You are a code reviewer. Provide constructive feedback on code.
        You must give atleast one feedback and get it improved before approving. 
        respond with 'APPROVED' if the content meets high standards.""",
        model_client_stream=True

    )

    termination_cond=TextMentionTermination("APPROVED")

    team = RoundRobinGroupChat([code_writer_agent,code_feedback_agent],
                               termination_condition=termination_cond)
    
    print("----REFLECTION PATTERN----")
    await Console(team.run_stream(task="""Write a python function for reversing a string. """))
    await model_client.close()

asyncio.run(termination())