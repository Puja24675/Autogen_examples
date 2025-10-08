# Using Gemini
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import List,Literal
import os
from dotenv import load_dotenv

load_dotenv()

class BookReview(BaseModel):
    title: str
    author: str
    genre: List[str]
    rating: int 
    sentiment: Literal["Positive","Negative","Neutral"]
    summary: str
    pros: List[str]
    cons:List[str]
    recommendation: str

async def main()->None:
    # Gemini 2.5 is OpenAi compatible
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        # Gemini 2.5 is not in default model list
        model_info=ModelInfo(vision=True, function_calling=True, 
                             json_output=True,family="unknown", structured_output=True),
        api_key=os.getenv("GO0GLE_API_KEY"))

    book_review_agent = AssistantAgent(
        name = "book_reviewer",
        model_client = model_client,
        system_message = """You are a professional book reviewer agent.You must analyze only given  
        book thoroughly and provide structured reviews.""",
        output_content_type = BookReview
    )

    book = input("Enter any book: ")

    print("----Book review----")
    book_result = await book_review_agent.run(task = f"Review {book} book.")

    if isinstance(book_result.messages[-1], StructuredMessage):
        review = book_result.messages[-1].content
        print(f"\n--Title--:{review.title}")
        print(f"\n--Author--:{review.author}")
        print(f"\n--Genre--:{', '.join(review.genre)}")
        print(f"\n--Rating--:{review.rating}/10")
        print(f"\n--Sentiment--:{review.sentiment}")
        print(f"\n--Summary--:{review.summary}")
        print(f"\n--Pros--:{', '.join(review.pros)}")
        print(f"\n--Cons--:{', '.join(review.cons)}")
        print(f"\n--Recommendation--:{review.recommendation}")

    await model_client.close()

asyncio.run(main())