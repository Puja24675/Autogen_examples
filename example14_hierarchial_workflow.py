import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main()->None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    support_triage=AssistantAgent(
        "Support_triage",
        model_client=model_client,
        system_message="""You are Support Triage.
        Classify incoming support tickets:
        Simple/FAQ -> level1_support
        Technical Issues -> level2_support
        Complex/Engineering -> level3_support
        Account/Billing -> billing_specialist
        Always route to the appropriate level and provide ticket summary.""",
        description = "Support Triage - routes tickets to appropriate support level"
        )

    level1_support=AssistantAgent(
        "Level1_support",
        model_client=model_client,
        system_message="""You are a Level 1 support. Handle basic questions and common issues.
        if the issue is beyond your scope, escalate to level 2 with 'ESCALATE_L2' 
        and full context. Try to resolve simple password resets, account questions and basic 
        trouble shooting first.""",
        description="Level 1 support-handles basic user questions."
    )

    level2_support=AssistantAgent(
        "Level2_support",
        model_client=model_client,
        system_message="""You are a Level 2 techinal support. Handle complex technical issues. 
        You have access to advanced diagonstic tools and can perform system checks.
        if the issue requires engineering involvement, escalte to Level 3 with 'ESCALATE_L3'.""",
        description="Level 2 support-handles technical troublshooting."
    )

    level3_support=AssistantAgent(
        "Level3_support",
        model_client=model_client,
        system_message="""You are a Level 3 Engineering support. Handle the most 
        complex technical issues. you can access system logs, modify configurations, 
        and corodinate with development teams. Provide detailed technical 
        analysis and permanent solutions.""",
        description="Level 3 support-handles engineering level issues."
    )

    billing_specialist=AssistantAgent(
        "billing_specialist",
        model_client=model_client,
        system_message="""You are a billing specialist. Hnadle all account, payment, and
        subscription issues. Youhave access to billing systems and can process refunds,
        upgrades, and account changes. Escalate only if legal or executive approval is needed.""",
        description="Billing Specialist-Handles account and payment issues."
    )

    support_manager=AssistantAgent(
        "support_manager",
        model_client=model_client,
        system_message="""You are a support manager.You oversee all support operations.
        You handle escalations from all levels, make policy decisions, and" coordinate
        with other depatments. You also handle VIP customers and complex multi-department issues.""",
        description="Support manager- supervises all support operations"
    )

    support_team=SelectorGroupChat(
        participants=[
            support_triage,level1_support,level2_support,level3_support,support_manager
        ],
        model_client=model_client,
        termination_condition=MaxMessageTermination(12),
        allow_repeated_speaker=True
    )

    print("-----MULTI-LEVEL SUPPORT HIERARCHY-----")

    support_tickets=[
        # """I forgot my password and can't log into my account. Can you help me rest it?""",
        # """My API calls are returning 500 errors.""",
        # """I need to upgrade ny subscription but billing page shows error.
        # Also can i get refund for last month's unused credits?""",
        """Our entire production system is down. databse connections are failing and we're losing 
        revenue every minute."""
    ]

    for i,ticket in enumerate(support_tickets,1):
        print(f"\n--Ticket number is --:{i}")
        print(f"\n--Customer ticket--:{ticket}")

        await Console(support_team.run_stream(task=f"Raised ticket:{ticket}"))

        if i<len(support_tickets):
            await support_team.reset()
            print("\n"+"+"*10)
    
    await model_client.close()

asyncio.run(main())