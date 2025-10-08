import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import Handoff
from autogen_agentchat.ui import Console
# from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main()->None:
    model_client=OpenAIChatCompletionClient(model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))


    requirement_analyst=AssistantAgent(
        name="requirement_analyst",
        model_client=model_client,
        handoffs=[Handoff(target="architect")],
        system_message="""you analyze requirements and create detailed specifications.
        When complete, hand off to the architect. """
    )

    architect=AssistantAgent(
        name="architect",
        model_client=model_client,
        handoffs=[Handoff(target="developer")],
        system_message="""you design system architecture based on requirements..
        When complete, hand off to the developer. """
    )

    developer=AssistantAgent(
        name="developer",
        model_client=model_client,
        handoffs=[Handoff(target="COMPLETE")],
        system_message="""You create implementation plans based on architecture.
         When finished, say 'TERMINATE'. """,
    )

    handoff_complete=HandoffTermination(target="COMPLETE")
    text_termination=TextMentionTermination("TERMINATE")

    team=RoundRobinGroupChat(
        [requirement_analyst,architect,developer],
        termination_condition=handoff_complete | text_termination
    )

    print("---SEQUENTIAL WORKFLOW WITH HANDOFFS---")
    await Console(team.run_stream(task="""Design and plan the implementation of a real
        time chat application with user authentication and message history."""))

    await model_client.close()

asyncio.run(main())
    